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

    - What to do with bad channels? and issues with labels from files with unclear names of result files?

    - COMMENT this code early enough!
    
    - Frequency analysis?
'''


def calculate_mean_decoding_accuracy():
    os.makedirs('data/{}'.format(DATE_TIME), exist_ok=True)

    subject_ids = np.array(get_subject_ids(os.listdir(epoched_data_path)))

    accuracies = []

    for subject_id in subject_ids:
        print("\n------------------------\nLOGGER: Decoding subject #{}\n------------------------\n".format(subject_id))
        epochs_list, labels_list = load_subject_train_data(subject_id)
        acc = decode_subject_response_over_time(epochs_list, labels_list, subject_id)
        accuracies.append(acc)

    accuracies = np.array(accuracies)



    np.save('results/mvpa_accuracies_{}.npy'.format(DATE_TIME), accuracies)

    # PLOT (could get its own method)

    # Create a seaborn lineplot, passing the matrix directly to seaborn
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size

    # Create the lineplot, seaborn will automatically calculate confidence intervals
    sns.lineplot(data=np.mean(accuracies, axis=0))

    # Set plot labels and title
    plt.xlabel('Time (samples)')
    plt.ylabel('Mean Activation')
    plt.title('Mean Activation with Confidence Intervals')

    plt.savefig('results/mean_accuracies_{}.png'.format(DATE_TIME))

    # Show the plot
    plt.show()


def load_results_and_plot(path: str) -> None:
    accuracies = np.load(path)
    # Create a seaborn lineplot, passing the matrix directly to seaborn
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size

    # Create the lineplot, seaborn will automatically calculate confidence intervals
    sns.lineplot(data=np.mean(accuracies, axis=0))

    # Set plot labels and title
    plt.xlabel('Time (samples)')
    plt.ylabel('Mean Activation')
    plt.title('Mean Activation with Confidence Intervals')

    # Show the plot
    plt.show()


def decode_subject_response_over_time(epochs_list: List[EpochsFIF],
                                      labels_list: List[List[int]],
                                      subject_id: int) -> List[float]:
    scaled_epochs = []
    labels_filtered = []

    # This loop rescales the data one experiment block at a time. This is because baseline channel activation
    # varies a lot between blocks, but channel activation varies on a much smaller scale inside a block.
    for epoch, labels in zip(epochs_list, labels_list):
        data = epoch.get_data()

        data_filtered = data[~np.isnan(labels)]
        labels_filtered.extend(labels[~np.isnan(labels)])

        scaler = StandardScaler()
        num_epochs, num_channels, num_timesteps = data_filtered.shape

        # Transpose and reshape for StandardScaler (only takes 2d input)
        data_filtered = np.transpose(data_filtered, axes=(0, 2, 1))
        data_filtered = np.reshape(data_filtered, shape=(-1, num_channels))

        data_scaled = scaler.fit_transform(data_filtered)

        # Reshape and transpose back
        data_scaled = np.reshape(data_scaled, shape=(num_epochs, num_timesteps, num_channels))
        data_scaled = np.transpose(data_scaled, axes=(0, 2, 1))

        scaled_epochs.append(data_scaled)

    scaled_epochs = np.concat(scaled_epochs, axis=0)
    labels_filtered = np.array(labels_filtered)

    np.save('data/{}/scaled_epochs_{}.npy'.format(DATE_TIME, subject_id), scaled_epochs)
    accuracies = []

    for t in range(scaled_epochs.shape[-1]):
        # clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, random_state=0, tol=1e-5))
        clf = svm.SVC(kernel='linear', C=1, random_state=42)
        scores = cross_val_score(clf, scaled_epochs[:, :, t], labels_filtered, cv=5)
        accuracies.append(np.mean(scores))

    return accuracies


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


if __name__ == '__main__':
    calculate_mean_decoding_accuracy()

    # load_results_and_plot('/Users/joyet/Documents/Documents - Guillaume’s MacBook Pro/UniBasel/MSc_Data_Science/Master Thesis/Code/eeg-master-thesis/mvpa/results/mvpa_accuracies.npy')
