import os.path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import cross_val_score
from datetime import datetime
import time
import timeit

import matplotlib

matplotlib.use('macOSX')
import matplotlib.pyplot as plt

from dataloader import get_subject_ids, load_subject_train_data, preprocess_train_data, average_augment_data

epoched_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/preprocessed/stim_epochs'
behav_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/raw'

# Get date and time for naming files (prevents overwriting)
now = datetime.now()
DATE_TIME = now.strftime("D%Y-%m-%d_T%H-%M-%S")
print("\n{}\n".format(DATE_TIME))

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


def calculate_mean_decoding_accuracy(test: bool = False):
    """
    Outermost method. Calls function to load, process and decode data (once per subject).
    :param test: if True, decodes only a small number of subjects.
    :return: None
    """
    subsampling_rate = 5
    pseudo_k = 4
    augment_factor = 2
    C = 2.5
    window_width = 1

    subject_ids = get_subject_ids(epoched_data_path)

    # 'test' being True makes the code run on only a few subjects
    if test:
        subject_ids = np.random.choice(subject_ids, 10)

    accuracies = []

    for subject_id in subject_ids:
        epochs_list, labels_list = load_subject_train_data(subject_id)
        proc_epochs = preprocess_train_data(epochs_list, subsampling_rate=subsampling_rate)
        proc_labels = np.concat(labels_list, axis=0)

        # TODO: debug this. Does it make sense to do this after PCA? (probably not)
        proc_epochs, proc_labels = average_augment_data(proc_epochs, proc_labels, pseudo_k=pseudo_k, augment_factor=augment_factor)

        print("\n---------------------------\nLOGGER: Decoding subject #{}\n---------------------------\n".format(
            subject_id))
        acc = decode_subject_response_over_time(proc_epochs, proc_labels, C=C, window_width=window_width)

        accuracies.append(acc)

    accuracies = np.array(accuracies)

    if not test:
        np.save('results/mvpa_acc_{}Hz_{}-av_{}-aug_{}-C_{}-win.png'.format(1000/subsampling_rate,
                                                                            pseudo_k,
                                                                            augment_factor,
                                                                            C*1000,
                                                                            window_width), accuracies)

    plot_accuracies(data=accuracies)


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

    for t in range(proc_epochs.shape[-1] - window_width + 1):
        clf = svm.SVC(kernel='linear', C=C)
        scores = cross_val_score(clf, np.reshape(proc_epochs[:, :, t:t+window_width], shape=(num_epochs, -1)),
                                 proc_labels, cv=5)
        subject_accuracies.append(np.mean(scores))

    return np.array(subject_accuracies)


def plot_accuracies(data: np.ndarray = None, path: str = None, subsampling_rate: int = 5,
                    pseudo_k: int = 4, augment_factor: int = 1, C: int = 1, window_width: int = 1) -> None:
    """
    Plots the mean accuracy over time with confidence band over subjects.
    :param data: 2D numpy array, where each row is the decoding accuracy for one subject over all timesteps.
    :param path: if data is None, this path indicates what file to load the data from.
    The parameters below are hyperparameters of the model passed to this method for naming the plot.
    :param subsampling_rate:
    :param pseudo_k:
    :param augment_factor:
    :param C:
    :param window_width:
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
    sns.lineplot(data=df, x='Time', y='Mean_Accuracy', errorbar='sd')

    # Set plot labels and title
    plt.xlabel('Time (samples)')
    plt.ylabel('Mean Accuracy')
    plt.title('{} Hz / {}-fold average / {}-fold augment / SVM: C = {}, window = {}'.format(1000/subsampling_rate,
                                                                                            pseudo_k,
                                                                                            augment_factor,
                                                                                            C,
                                                                                            window_width))

    if path is None:
        plt.savefig('results/mean_acc_{}Hz_{}-av_{}-aug_{}-C_{}-win.png'.format(1000/subsampling_rate,
                                                                                pseudo_k,
                                                                                augment_factor,
                                                                                C*1000,
                                                                                window_width))

    # Show the plot
    plt.show()


if __name__ == '__main__':
    calculate_mean_decoding_accuracy(test=False)

    # plot_accuracies(path='/Users/joyet/Documents/Documents - Guillaume’s MacBook Pro/UniBasel/MSc_Data_Science/Master Thesis/Code/eeg-master-thesis/mvpa/results/data/mvpa_accuracies_D2024-10-07_T14-26-14.npy')
    # plot_accuracies(path='/Users/joyet/Documents/Documents - Guillaume’s MacBook Pro/UniBasel/MSc_Data_Science/Master Thesis/Code/eeg-master-thesis/mvpa/results/data/mvpa_accuracies_D2024-10-09_T17-06-55.npy')
    # plot_accuracies(path='/Users/joyet/Documents/Documents - Guillaume’s MacBook Pro/UniBasel/MSc_Data_Science/Master Thesis/Code/eeg-master-thesis/mvpa/results/data/mvpa_accuracies_D2024-10-09_T18-06-23.npy')
