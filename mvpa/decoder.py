import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import cross_val_score

import matplotlib

matplotlib.use('macOSX')
import matplotlib.pyplot as plt

from utils.dataloader import get_subject_ids, load_subject_train_data, preprocess_train_data, average_augment_data


epoch_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/preprocessed/stim_epochs'
behav_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/raw'


'''
TODOS:
    - NEXT STEP: 
                + try augmentation with SVM with higher C.
                + try trial averaging before scaling/PCA.
    - PROBLEM WITH HIGH ACCURACY BEFORE STIMULUS ONSET: 
                + try stronger regularisation to prevent overfitting?
                + try using only certain brain areas (also to prevent overfitting) ?
                + try computing scaling and PCA only on training set, 
                    then apply it to validation set, during cross-validation.
    
    - What to do with bad channels? and issues with labels from files with unclear names of result files?

    - Frequency analysis?
'''


def calculate_mean_decoding_accuracy():
    """
    Outermost method. Calls function to load, process and decode data (once per subject).
    Arguments are passed from command line.
    :return: None
    """
    test = args.test
    downsample_factor = args.downsample_factor
    perform_PCA = args.perform_PCA
    pseudo_k = args.pseudo_k
    augment_factor = args.augment_factor
    C = args.SVM_C
    window_width = args.SVM_window_width

    print("\n--------------------------------------------------------------------------------\n",
          "LOGGER: STARTING RUN WITH PARAMETERS:\n",
          "{} Hz / {}-fold average / {}-fold augment / SVM: C = {}, window = {} / PCA: {}\n".format(
              1000 / downsample_factor,
              pseudo_k,
              augment_factor,
              C,
              window_width,
              "Yes" if perform_PCA else "No"),
          "--------------------------------------------------------------------------------\n", sep="")

    subject_ids = get_subject_ids(epoch_data_path)

    # 'test' being True makes the code run on only a few subjects
    if test:
        subject_ids = np.random.choice(subject_ids, 10)

    accuracies = []

    # Loop loads, processes and decodes data one subject at a time
    for subject_id in subject_ids:
        epochs, labels = load_subject_train_data(subject_id,
                                                 epoch_data_path=epoch_data_path,
                                                 behav_data_path=behav_data_path)

        proc_epochs = preprocess_train_data(epochs, downsample_factor=downsample_factor, perform_PCA=perform_PCA)

        if pseudo_k > 1:
            proc_epochs, labels = average_augment_data(proc_epochs, labels,
                                                       pseudo_k=pseudo_k,
                                                       augment_factor=augment_factor)

        print("\n-----------------------------\n",
              "LOGGER: Decoding subject #{:03d}\n".format(subject_id),
              "-----------------------------\n", sep="")

        acc = decode_subject_response_over_time(proc_epochs, labels, C=C, window_width=window_width)

        accuracies.append(acc)

    accuracies = np.array(accuracies)

    if not test:
        np.save('results/data/mvpa_acc_{}Hz_av-{}_aug-{}_C-{}_win-{}{}.npy'.format(int(1000 / downsample_factor),
                                                                                   pseudo_k,
                                                                                   augment_factor,
                                                                                   int(C * 1000),
                                                                                   window_width,
                                                                                   "_PCA" if perform_PCA else ""),
                accuracies)

    plot_accuracies(data=accuracies, downsample_factor=downsample_factor, pseudo_k=pseudo_k,
                    augment_factor=augment_factor, C=C, window_width=window_width, perform_PCA=perform_PCA)


def decode_subject_response_over_time(proc_epochs: np.ndarray[float],
                                      proc_labels: np.ndarray[int],
                                      C: int = 1,
                                      window_width: int = 1) -> np.ndarray[float]:
    """
    Decodes data from one subject.
    :param proc_epochs: numpy array already processed epochs (shape #epochs x #channels x #timesteps).
    :param proc_labels: numpy array of labels (length #epochs).
    :param C: regularisation parameter of SVM.
    :param window_width: width of sliding window (over time) that is fed as input to SVM.
    :return: numpy array of decoding accuracies, one accuracy per time point in the data.
    """
    subject_accuracies = []

    num_epochs = proc_epochs.shape[0]

    # Loop trains SVM for each timestep / window in the epoched data
    for t in range(proc_epochs.shape[-1] - window_width + 1):
        clf = svm.SVC(kernel='linear', C=C, class_weight='balanced')
        scores = cross_val_score(clf, np.reshape(proc_epochs[:, :, t:t + window_width], shape=(num_epochs, -1)),
                                 proc_labels, cv=5)
        subject_accuracies.append(np.mean(scores))

    return np.array(subject_accuracies)


def plot_accuracies(data: np.ndarray = None, path: str = None, downsample_factor: int = 5,
                    pseudo_k: int = 4, augment_factor: int = 1, C: int = 1,
                    window_width: int = 1, perform_PCA: bool = False) -> None:
    """
    Plots the mean accuracy over time with confidence band over subjects.
    :param data: 2D numpy array, where each row is the decoding accuracy for one subject over all timesteps.
    :param path: if data is None, this path indicates what file to load the data from.
    The parameters below are hyperparameters of the model passed to this method for naming the plot.
    :param downsample_factor:
    :param pseudo_k:
    :param augment_factor:
    :param C:
    :param window_width:
    :param perform_PCA:
    :return: None
    """
    if data is None:
        data = np.load(path)

    df = pd.DataFrame(data=data.T)
    df = df.reset_index().rename(columns={'index': 'Time'})
    df = df.melt(id_vars=['Time'], value_name='Mean_Accuracy', var_name='Subject')

    # Create a seaborn lineplot, passing the matrix directly to seaborn
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size

    # Create the lineplot, seaborn will automatically calculate confidence intervals
    sns.lineplot(data=df, x=df['Time'] * downsample_factor - 1000, y='Mean_Accuracy', errorbar='sd', label='Accuracy')
    sns.despine()

    plt.axhline(y=0.5, color='orange', linestyle='dashdot', linewidth=1, label='Random Chance')
    plt.axvline(x=0, ymin=0, ymax=0.05, color='black', linewidth=1, label='Stimulus Onset')

    # Set plot labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('{} Hz / {}-fold average / {}-fold augment / SVM: C = {}, window = {} / PCA: {}'.format(
        int(1000 / downsample_factor),
        pseudo_k,
        augment_factor,
        C,
        window_width,
        "Yes" if perform_PCA else "No"))

    if path is None:
        plt.savefig('results/mean_acc_{}Hz_av-{}_aug-{}_C-{}_win-{}{}.png'.format(int(1000 / downsample_factor),
                                                                                  pseudo_k,
                                                                                  augment_factor,
                                                                                  int(C * 1000),
                                                                                  window_width,
                                                                                  "_PCA" if perform_PCA else ""))

    # Show the plot
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments

    parser.add_argument('--test',
                        action='store_true',
                        help='Activate test mode (only few subjects are decoded for shorter runtime).')

    parser.add_argument('--downsample_factor',
                        default=5,
                        type=int,
                        required=False,
                        help='Number of timesteps that are collapsed into one when downsampling.')

    parser.add_argument('--perform_PCA',
                        action='store_true',
                        help='Perform PCA before decoding')

    parser.add_argument('--pseudo_k',
                        default=1,
                        type=int,
                        required=False,
                        help='Number of trials that are averaged. If 1, no trial averaging is performed.')

    parser.add_argument('--augment_factor',
                        default=1,
                        type=int,
                        required=False,
                        help='Factor with which the data is augmented (only matters when pseudo_k > 1).')

    parser.add_argument('--SVM_C',
                        default=1,
                        type=float,
                        required=False,
                        help='Regularisation parameter of the SVM.')

    parser.add_argument('--SVM_window_width',
                        default=1,
                        type=int,
                        required=False,
                        help='Width of time window that is fed as input to the SVM.')

    args = parser.parse_args()

    calculate_mean_decoding_accuracy()
