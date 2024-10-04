import os.path
import re
from typing import Dict, Tuple, Union, List

import mne
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
from datetime import datetime
import time
import timeit

import matplotlib
from mne.epochs import EpochsFIF

matplotlib.use('macOSX')
import matplotlib.pyplot as plt

epoched_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/preprocessed/stim_epochs'
behav_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/raw'

# Get date and time for naming files (prevents overwriting)
now = datetime.now()
DATE_TIME = now.strftime("D%Y-%m-%d_T%H-%M-%S")

'''
TODOS:
    - Most pressing issue: channels only vary at very small scales inside a block, but a lot between blocks. 
        How do still make an SVM work? Scaling? Can PCA also help?
        -> I need to scale each block individually I think. Therefore do not concatenate epochs and scale them
           at the beginning of decode_subject_response_over_time().
        -> Better for now, but maybe I should even scale each channel separately?
        -> TO TEST:  Scaler w/o PCA (now), PCA w/o Scaler, Scaler then PCA, PCA then Scaler
        -> OR TEST: normalise each channel for each trial by first subtracting the mean of the activation from
                    t = -1000ms to t = 0ms, then scale.
        -> FIRST: Subsampling.

    - What to do with bad channels? and issues with labels from files with unclear names of result files?

    - COMMENT this code early enough!
    
    - Frequency analysis?
'''


def calculate_mean_decoding_accuracy(test: bool = False):
    if not test:
        os.makedirs('data/{}'.format(DATE_TIME), exist_ok=True)

    subject_ids = np.array(get_subject_ids(os.listdir(epoched_data_path)))

    accuracies = []

    for subject_id in subject_ids:
        epochs_list, labels_list = load_subject_train_data(subject_id)
        proc_epochs, proc_labels = preprocess_train_data(epochs_list, labels_list, subject_id,
                                                         subsampling_rate=5, test=test)
        print("\n---------------------------\nLOGGER: Decoding subject #{}\n---------------------------\n".format(
            subject_id))
        acc = decode_subject_response_over_time(proc_epochs, proc_labels)

        accuracies.append(acc)

        # 'test' variable breaks out of loop early so that the code runs faster
        if test:
            if subject_id >= 10:
                break

    accuracies = np.array(accuracies)

    if not test:
        np.save('results/mvpa_accuracies_{}.npy'.format(DATE_TIME), accuracies)

    plot_accuracies(data=accuracies)


def decode_subject_response_over_time(proc_epochs: np.ndarray, proc_labels: np.ndarray) -> np.ndarray:
    subject_accuracies = []

    for t in range(proc_epochs.shape[-1]):
        clf = svm.SVC(kernel='linear', C=1, random_state=42)
        scores = cross_val_score(clf, proc_epochs[:, :, t], proc_labels, cv=5)
        subject_accuracies.append(np.mean(scores))

    return np.array(subject_accuracies)


def preprocess_train_data(epochs_list: List[EpochsFIF],
                          labels_list: List[List[int]],
                          subject_id: int,
                          subsampling_rate: int = 1,
                          test: bool = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    For now things like scaling and PCA are calculated on complete data.
    If we want to calculate it on train and apply it on test, we will need to do it differently
    (since for now train/test splits are done inside cross_val_scores).
    """
    proc_epochs = []
    labels_filtered = []

    # This loop rescales the data one experiment block at a time. This is because baseline channel activation
    # varies a lot between blocks, but channel activation varies on a much smaller scale inside a block.
    for epochs, labels in zip(epochs_list, labels_list):
        data = epochs.get_data()

        data = data[~np.isnan(labels)]
        labels_filtered.extend(labels[~np.isnan(labels)])

        num_epochs, num_channels, num_timesteps = data.shape

        # SUBSAMPLE from 1000 Hz to (1000 / subsampling_rate) Hz
        # Ignore first time-step since there are 2251 time-steps, which is not easily divisible.
        data = np.reshape(data[:, :, 1:], shape=(num_epochs, num_channels, num_timesteps // subsampling_rate,
                                                 subsampling_rate))
        data = np.mean(data, axis=-1)
        num_timesteps = data.shape[-1]

        # SUBTRACT MEAN from the 1000ms before stimulus presentation as a baseline
        data -= np.mean(data[:, :, :1000 // subsampling_rate], axis=2)[:, :, np.newaxis]

        # Transpose and for StandardScaler (expects features (channels) at last dimension)
        data = np.transpose(data, axes=(0, 2, 1))
        data = np.reshape(data, shape=(num_epochs * num_timesteps, num_channels))

        # SCALE
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        # Transpose back
        data = np.reshape(data, shape=(num_epochs, num_timesteps, num_channels))
        data = np.transpose(data, axes=(0, 2, 1))

        proc_epochs.append(data)

    proc_epochs = np.concat(proc_epochs, axis=0)
    labels_filtered = np.array(labels_filtered)

    if not test:
        np.save('data/{}/processed_epochs_{}.npy'.format(DATE_TIME, subject_id), proc_epochs)

    return proc_epochs, labels_filtered


def load_subject_train_data(subject_id: int) -> Tuple[List[EpochsFIF], List[np.ndarray[int]]]:
    epochs_dict = load_subject_epochs(subject_id)
    results_df = load_subject_labels(subject_id)

    epochs_list = []
    labels_list = []

    for block in epochs_dict.keys():
        epochs = epochs_dict[block]
        labels = (results_df[results_df['run'] == block]['response']).reset_index(drop=True)

        selected_labels = labels.loc[epochs.selection]

        epochs_list.append(epochs)
        labels_list.append(np.array(selected_labels))

    return epochs_list, labels_list


def load_subject_epochs(subject_id: int) -> Dict[int, EpochsFIF]:
    """
    :return: a dict with block numbers as keys and epochs of the corresponding blocks as values.
    """
    filenames = os.listdir(epoched_data_path)

    all_epochs = {}

    for fname in filter(lambda k: '_sj{}_'.format(subject_id) in k, filenames):
        epochs = mne.read_epochs(os.path.join(epoched_data_path, fname))
        pattern = r"_block(\d)_"
        match = re.search(pattern, fname)
        block = int(match.group(1))
        all_epochs[block] = epochs

    return all_epochs


def load_subject_labels(subject_id: int) -> pd.DataFrame:
    subdirectory_content = os.listdir(os.path.join(behav_data_path, str(subject_id)))

    cols = ['session', 'run', 'response', 'confidence', 'correct']
    dfs = []

    # TODO: correct criteria for .csv selection
    for filename in filter(lambda k: ('results.csv' in k and 'assr' not in k and 'wrong' not in k),
                           subdirectory_content):
        data = pd.read_csv(os.path.join(behav_data_path, str(subject_id), filename), usecols=cols)
        dfs.append(data)

    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df


def get_subject_ids(filenames: List[str]) -> List[int]:
    subject_ids = []

    # Regular expression pattern to match "_sj" followed by digits and ending with "_"
    pattern = r"_sj(\d+)_"

    # Loop through each string in the list
    for fn in filenames:
        match = re.search(pattern, fn)
        if match:
            # Extract the digits (group 1) and add to the list
            subject_ids.append(int(match.group(1)))

    # eliminate duplicates and sort
    subject_ids = list(set(subject_ids))
    subject_ids.sort()

    # The removals below are just to allow testing: the issues need to be fixed/understood.

    # TODO: check what is up with those result files (several blocks with same number)
    subject_ids.remove(9)
    subject_ids.remove(18)
    subject_ids.remove(34)
    subject_ids.remove(117)

    # TODO: subjects with discrepant bad channels
    # subject_ids.remove(13)
    # subject_ids.remove(24)
    # subject_ids.remove(25)
    # subject_ids.remove(40)
    # subject_ids.remove(124)
    # subject_ids.remove(137)

    # Print the extracted numbers
    return subject_ids


def plot_accuracies(data: np.ndarray = None, path: str = None) -> None:
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
    calculate_mean_decoding_accuracy()

    # plot_accuracies(path='/Users/joyet/Documents/Documents - Guillaume’s MacBook Pro/UniBasel/MSc_Data_Science/Master Thesis/Code/eeg-master-thesis/mvpa/results/mvpa_accuracies_D2024-10-03_T16-50-01.npy')
