import os.path
import re
from typing import Dict, Tuple

import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from mne.epochs import EpochsFIF


def average_augment_data(epochs: np.ndarray[float],
                         labels: np.ndarray[int],
                         pseudo_k: int = 4,
                         augment_factor: int = 1) -> Tuple[np.ndarray[float], np.ndarray[int]]:
    """
    Randomly averages trials with same label to create pseudo-trials. Optionally augments data by averaging the
    data several times in different partitions, i.e. each trials is used for several pseudo-trials.
    :param epochs: numpy array of original epoch data (of shape #epochs x #timesteps x #channels).
    :param labels: numpy array of original labels.
    :param pseudo_k: number of trials to be averaged per pseudo-trial.
    :param augment_factor: number of pseudo-trials each original trial will be in.
    :return: numpy array with pseudo-trials (of shape #pseudo-epochs x #timesteps x #channels),
             numpy array with corresponding labels.
    """
    # TODO: debug this.

    pseudo_epochs = []
    pseudo_labels = []

    for lab in [-1, 1]:
        side_epochs = epochs[labels == lab]

        # n is the number of partitions, and k is the number of samples per partition
        n = side_epochs.shape[0] // pseudo_k

        for _ in range(augment_factor):
            # Step 1: Shuffle the data along the first dimension
            shuffled_array = np.random.permutation(side_epochs)

            # Step 2: Split the shuffled array into n partitions
            partitions = np.array_split(shuffled_array, n)
            partition_averages = np.stack([np.mean(p, axis=0) for p in partitions])

            # Step 3: Take the mean over epochs in each partition
            pseudo_epochs.append(partition_averages)
            pseudo_labels.extend(np.full(partition_averages.shape[0], lab))

    pseudo_epochs = np.concat(pseudo_epochs, axis=0)
    pseudo_labels = np.array(pseudo_labels)

    # Reshuffle data
    shuffle_idxs = np.random.permutation(range(len(pseudo_labels)))
    pseudo_epochs = pseudo_epochs[shuffle_idxs]
    pseudo_labels = pseudo_labels[shuffle_idxs]

    return pseudo_epochs, pseudo_labels


def load_subject_train_data(subject_id: int,
                            epoch_data_path: str,
                            behav_data_path: str,
                            downsample_factor: int = 1) -> Tuple[np.ndarray[float], np.ndarray[int]]:
    """
    Loads and correctly combines epoch data and labels (behavioural outcomes) for one subject.
    Applies baseline, drops NaN labels, downsamples.
    :param subject_id:
    :param epoch_data_path: path to epoch data.
    :param behav_data_path: path to behavioural data. Expects data of each subject to be in a folder named <subject_id>.
    :param downsample_factor: number of samples that are collapsed into one by averaging.
    :return: numpy array containing epoch data (of shape #epochs x #timesteps x #channels),
             numpy array with corresponding labels (of length #epochs).
    """
    epochs_dict = load_subject_epochs(epoch_data_path, subject_id)
    results_df = load_subject_labels(behav_data_path, subject_id)

    epochs_list = []
    labels_list = []

    for block in epochs_dict.keys():
        epochs = epochs_dict[block]
        epochs = epochs.apply_baseline((-1.000, -0.001))
        labels = (results_df[results_df['run'] == block]['response']).reset_index(drop=True)

        # select labels for which the corresponding epoch was accepted
        selected_labels = np.array(labels.loc[epochs.selection])

        data = epochs.get_data()

        # drop NaN labels (no response) and corresponding epochs
        epochs_list.append(data[~np.isnan(selected_labels)])
        labels_list.append(selected_labels[~np.isnan(selected_labels)])

    epochs = np.concat(epochs_list, axis=0)
    labels = np.concat(labels_list, axis=0)

    num_epochs, num_channels, num_timesteps = epochs.shape

    # DOWNSAMPLE from 1000 Hz to (1000 / downsample_factor) Hz
    # Ignore first time-step since there are 2251 time-steps, which is not easily divisible.
    if downsample_factor > 1:
        epochs = np.reshape(epochs[:, :, 1:], shape=(num_epochs, num_channels, num_timesteps // downsample_factor,
                                                     downsample_factor))
        epochs = np.mean(epochs, axis=-1)

    # Transpose to get shape (#epochs x #timesteps x #channels)
    epochs = np.transpose(epochs, axes=(0, 2, 1))

    return epochs, labels


def load_subject_epochs(path: str, subject_id: int) -> Dict[int, EpochsFIF]:
    """
    Loads epoch data for one subject.
    :param path: path to epoch data.
    :param subject_id:
    :return: dict, keys are experiment block numbers, values are a mne.EpochsFIF object for the corresponding block.
    """
    filenames = os.listdir(path)

    all_epochs = {}

    for fname in filter(lambda k: '_sj{}_'.format(subject_id) in k, filenames):
        epochs = mne.read_epochs(os.path.join(path, fname))
        pattern = r"_block(\d)_"
        match = re.search(pattern, fname)
        block = int(match.group(1))
        all_epochs[block] = epochs

    return all_epochs


def load_subject_labels(path: str, subject_id: int) -> pd.DataFrame:
    """
    Loads labels (behavioural outcomes) for one subject.
    :param path: path to behavioural data. Expects data of each subject to be in a folder named <subject_id>.
    :param subject_id:
    :return: a pandas dataframe containing subject response and
             block number ('run') for subsequent join with epoch data.
    """
    subdirectory_content = os.listdir(os.path.join(path, str(subject_id)))

    cols = ['session', 'run', 'response', 'confidence', 'correct']
    dfs = []

    # TODO: correct criteria for .csv selection
    for filename in filter(lambda k: ('{}_'.format(subject_id) in k and
                                      'results.csv' in k and
                                      'assr' not in k and
                                      'wrong' not in k),
                           subdirectory_content):
        data = pd.read_csv(os.path.join(path, str(subject_id), filename), usecols=cols)
        dfs.append(data)

    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df


def get_subject_ids(path: str) -> np.ndarray:
    """
    Scan through directory to get a list of subject IDs.
    :param path: path to the directory containing the files.
    :return: numpy array of subject IDs.
    """
    filenames = os.listdir(path)

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

    # The removals below are just to allow testing: the issues need to be fixed.

    # TODO: check what is up with those result files (several blocks with same number)
    try:
        subject_ids.remove(9)
        subject_ids.remove(18)
        subject_ids.remove(34)
        subject_ids.remove(117)
    except ValueError:
        pass

    # TODO: subjects with discrepant bad channels
    # subject_ids.remove(13)
    # subject_ids.remove(24)
    # subject_ids.remove(25)
    # subject_ids.remove(40)
    # subject_ids.remove(124)
    # subject_ids.remove(137)

    return np.array(subject_ids)
