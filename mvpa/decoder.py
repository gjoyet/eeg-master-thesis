import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score

import matplotlib

matplotlib.use('macOSX')
import matplotlib.pyplot as plt

from utils.dataloader import get_subject_ids, load_subject_train_data, average_augment_data

epoch_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/preprocessed/stim_epochs'
behav_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/raw'

'''
TODOS:
    - What to do with bad channels? and issues with labels from files with unclear names of result files?
        -> have a look at data with David.

    - Frequency analysis?
'''


def calculate_mean_decoding_accuracy():
    """
    Outermost method. Calls function to load, process and decode data (once per subject).
    Arguments are passed from command line.
    :return: None
    """
    downsample_factor = args.downsample_factor
    pseudo_k = args.pseudo_k
    augment_factor = args.augment_factor
    C = args.SVM_C
    window_width = args.SVM_window_width

    title = '{} Hz / {}-fold average / {}-fold augment / SVM: C = {}, window = {}'.format(
        int(1000 / downsample_factor),
        pseudo_k,
        augment_factor,
        C,
        window_width)

    filename = 'accuracy_{}Hz_av-{}_aug-{}_C-{}_win-{}'.format(int(1000 / downsample_factor),
                                                               pseudo_k,
                                                               augment_factor,
                                                               int(C * 1000),
                                                               window_width)

    print("\n--------------------------------------------------------------------------------\n",
          "LOGGER: STARTING RUN WITH PARAMETERS:\n",
          title,
          "--------------------------------------------------------------------------------\n", sep="")

    subject_ids = get_subject_ids(epoch_data_path)

    accuracies = []

    # Loop loads, processes and decodes data one subject at a time
    for subject_id in subject_ids:
        epochs, labels = load_subject_train_data(subject_id,
                                                 epoch_data_path=epoch_data_path,
                                                 behav_data_path=behav_data_path,
                                                 downsample_factor=downsample_factor)

        if pseudo_k > 1:
            epochs, labels = average_augment_data(epochs, labels,
                                                  pseudo_k=pseudo_k,
                                                  augment_factor=augment_factor)

        print("\n-----------------------------\n",
              "LOGGER: Decoding subject #{:03d}\n".format(subject_id),
              "-----------------------------\n", sep="")

        acc = decode_response_over_time(epochs, labels, C=C, window_width=window_width)

        accuracies.append(acc)

    accuracies = np.array(accuracies)

    np.save('results/data/{}.npy'.format(filename), accuracies)

    plot_accuracies(data=accuracies, title=title, savefile=filename)


def decode_response_over_time(epochs: np.ndarray[float],
                              labels: np.ndarray[int],
                              C: int = 1,
                              window_width: int = 1) -> np.ndarray[float]:
    """
    Decodes participant's response from epoch data.
    :param epochs: numpy array of epoch data (shape #epochs x #timesteps x #channels).
    :param labels: numpy array of labels (length #epochs).
    :param C: regularisation parameter of SVM.
    :param window_width: width of sliding window (over time) that is fed as input to SVM.
    :return: numpy array of decoding accuracies, one accuracy per time point in the data.
    """
    subject_accuracies = []

    num_epochs = epochs.shape[0]

    # Loop trains SVM for each timestep / window in the epoch data
    for t in range(epochs.shape[1] - window_width + 1):
        pipeline = Pipeline([('scaler', StandardScaler()),
                             ('svc', svm.SVC(kernel='linear', C=C, class_weight='balanced'))])
        scores = cross_val_score(pipeline, np.reshape(epochs[:, t:t + window_width, :], shape=(num_epochs, -1)),
                                 labels, cv=5)
        subject_accuracies.append(np.mean(scores))

    return np.array(subject_accuracies)


def plot_accuracies(data: np.ndarray = None, title: str = "", savefile: str = None,
                    downsample_factor: int = 5) -> None:
    """
    Plots the mean accuracy over time with confidence band over subjects.
    :param data: 2D numpy array, where each row is the decoding accuracy for one subject over all timesteps.
    :param title: title of the plot.
    :param savefile: file name to save the plot under. If None, no plot is saved.
    :param downsample_factor:
    :return: None
    """

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
    plt.title(title)

    if savefile is not None:
        plt.savefig('results/{}.png'.format(savefile))

    # Show the plot
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments

    parser.add_argument('--downsample_factor',
                        default=5,
                        type=int,
                        required=False,
                        help='Number of timesteps that are collapsed into one when downsampling.')

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
