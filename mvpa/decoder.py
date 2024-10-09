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

from dataloader import get_subject_ids, load_subject_train_data, preprocess_train_data

epoched_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/preprocessed/stim_epochs'
behav_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/raw'

# Get date and time for naming files (prevents overwriting)
now = datetime.now()
DATE_TIME = now.strftime("D%Y-%m-%d_T%H-%M-%S")

'''
TODOS:
    - FIRST: move certain methods to decoder.py
    - NEXT STEP: augment data (before or after scaling/PCA?).
    - What to do with bad channels? and issues with labels from files with unclear names of result files?

    - Frequency analysis?
'''


def calculate_mean_decoding_accuracy(test: bool = False):
    """
    Outermost method. Calls function to load, process and decode data (once per subject).
    :param test: if True, does not store processed data and decodes only a small number of subjects.
    :return: None
    """
    if not test:
        os.makedirs('data/{}'.format(DATE_TIME), exist_ok=True)

    subject_ids = get_subject_ids(epoched_data_path)

    # 'test' being True makes the code run on only a few subjects
    if test:
        subject_ids = np.random.choice(subject_ids, 8)

    accuracies = []

    for subject_id in subject_ids:
        epochs_list, labels_list = load_subject_train_data(subject_id)
        proc_epochs = preprocess_train_data(epochs_list, subsampling_rate=5)
        proc_labels = np.concat(labels_list, axis=0)
        if not test:
            np.save('data/{}/processed_epochs_{}.npy'.format(DATE_TIME, subject_id), proc_epochs)

        print("\n---------------------------\nLOGGER: Decoding subject #{}\n---------------------------\n".format(
            subject_id))
        acc = decode_subject_response_over_time(proc_epochs, proc_labels)

        accuracies.append(acc)

    accuracies = np.array(accuracies)

    if not test:
        np.save('results/mvpa_accuracies_{}.npy'.format(DATE_TIME), accuracies)

    plot_accuracies(data=accuracies)


def decode_subject_response_over_time(proc_epochs: np.ndarray[float],
                                      proc_labels: np.ndarray[int]) -> np.ndarray[float]:
    """
    Decodes data from one subject.
    :param proc_epochs: numpy array already processed epochs (shape #epochs x #channels x #timesteps).
    :param proc_labels: numpy array of labels (length #epochs).
    :return: numpy array of decoding accuracies, one accuracy per time point in the data.
    """
    subject_accuracies = []

    for t in range(proc_epochs.shape[-1]):
        clf = svm.SVC(kernel='linear', C=1)
        scores = cross_val_score(clf, proc_epochs[:, :, t], proc_labels, cv=5)
        subject_accuracies.append(np.mean(scores))

    return np.array(subject_accuracies)


def plot_accuracies(data: np.ndarray = None, path: str = None) -> None:
    """
    Plots the mean accuracy over time with confidence band over subjects.
    :param data: 2D numpy array, where each row is the decoding accuracy for one subject over all timesteps.
    :param path: if data is None, this path indicates what file to load the data from.
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
    plt.title('Mean Accuracy with Confidence Intervals')

    if path is None:
        plt.savefig('results/mean_accuracies_{}.png'.format(DATE_TIME))

    # Show the plot
    plt.show()


if __name__ == '__main__':
    calculate_mean_decoding_accuracy(test=False)

    # plot_accuracies(path='/Users/joyet/Documents/Documents - Guillaume’s MacBook Pro/UniBasel/MSc_Data_Science/Master Thesis/Code/eeg-master-thesis/mvpa/results/mvpa_accuracies_D2024-10-03_T16-50-01.npy')
