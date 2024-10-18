import os.path
import re
from typing import Dict, Tuple, List

import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

from mne.epochs import EpochsFIF


# Subject IDs are inferred from epoch_data_path. Code in this file expects behavioural_data_path to contain
# data of each subject inside a folder named <subject_id>.
epoch_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/preprocessed/stim_epochs'
behavioural_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/raw'


'''
TODOS:
    - Change file such that only dataloader.py needs path variables for the data 
      instead of the files higher in the hierarchy.
'''


class CustomNPZDataset(Dataset):
    def __init__(self, file_path, transform=None):
        # Load the .npz file in 'mmap_mode' for memory-efficient access
        self.data = np.load(file_path, mmap_mode='r')

        # Assume the .npz file contains two arrays: 'inputs' and 'labels'
        self.inputs = self.data['epochs']
        self.labels = self.data['labels']
        self.transform = transform

    def __len__(self):
        return self.inputs.shape[0]  # Return the number of samples (rows)

    def __getitem__(self, idx):
        # Load a single input and label
        input_data = self.inputs[idx]
        label = self.labels[idx]

        # Apply any transformations if necessary
        if self.transform:
            input_data = self.transform(input_data)

        # Convert to PyTorch tensors and return
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def get_pytorch_dataloader(subject_ids: List[int],
                           downsample_factor: int = 1,
                           shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Returns a pytorch dataloader with the complete training data.
    Data have applied baseline, dropped NaN labels, and have been downsampled.
    Additionally, the data is scaled, which is not done when loading data of subjects individually. This is because
    the data should be scaled before it is put into a pytorch dataset object. The individual loading of subject data
    on the other hand is used by MVPA, where sklearn scales the data for each cross-validation fold separately, which
    is why it should not be scaled beforehand.
    :param subject_ids:
    :param downsample_factor: number of samples that are collapsed into one by averaging.
    :param shuffle: if True, training data is shuffled (as to prevent all epochs from the same subject being together).
    :return: pytorch dataloader with training data.
    """
    filename = '../../data/training_data_{}Hz.npz'.format(int(1000 / downsample_factor))

    # if file already exists, do nothing
    if not os.path.isfile(filename):
        # else load whole training data and save it in .npz file
        epoch_list = []
        label_list = []

        for sid in subject_ids:
            epochs, labels = load_subject_train_data(sid, epoch_data_path, behavioural_data_path,
                                                     downsample_factor=downsample_factor)
            epoch_list.append(epochs)
            label_list.append(labels)

        epochs = np.concat(epoch_list, axis=0)
        labels = np.concat(label_list, axis=0)
        labels = (labels + 1) / 2  # transform -1 labels to 0 (since we use BCELoss later)

        # SCALE
        scaler = StandardScaler()
        num_e, num_t, num_c = epochs.shape
        epochs = np.reshape(scaler.fit_transform(np.reshape(epochs, shape=(num_e * num_t, num_c))),
                            shape=(num_e, num_t, num_c))

        if shuffle:
            shuffle_idxs = np.random.permutation(range(len(labels)))
            epochs = epochs[shuffle_idxs]
            labels = labels[shuffle_idxs]

        np.savez(filename, epochs=epochs, labels=labels)

    # Define dataset and DataLoader
    dataset = CustomNPZDataset(file_path=filename)

    # Use DataLoader for batch loading
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # test num_workers = 1, 2, 4, ...

    return dataloader


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

    # Reshuffle data (so not all data with same label are together)
    shuffle_idxs = np.random.permutation(range(len(pseudo_labels)))
    pseudo_epochs = pseudo_epochs[shuffle_idxs]
    pseudo_labels = pseudo_labels[shuffle_idxs]

    return pseudo_epochs, pseudo_labels


def load_subject_train_data(subject_id: int,
                            downsample_factor: int = 1) -> Tuple[np.ndarray[float], np.ndarray[int]]:
    """
    Loads and correctly combines epoch data and labels (behavioural outcomes) for one subject.
    Applies baseline, drops NaN labels, downsamples.
    :param subject_id:
    :param downsample_factor: number of samples that are collapsed into one by averaging.
    :return: numpy array containing epoch data (of shape #epochs x #timesteps x #channels),
             numpy array with corresponding labels (of length #epochs).
    """
    epochs_dict = load_subject_epochs(epoch_data_path, subject_id)
    results_df = load_subject_labels(behavioural_data_path, subject_id)

    epochs_list = []
    labels_list = []

    for block in epochs_dict.keys():
        epochs = epochs_dict[block]
        epochs = epochs.apply_baseline((-1.000, -0.001), verbose=False)
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
        epochs = mne.read_epochs(os.path.join(path, fname), verbose=False)
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


def get_subject_ids() -> np.ndarray:
    """
    Scan through directory to get a list of subject IDs.
    :return: numpy array of subject IDs.
    """
    filenames = os.listdir(epoch_data_path)

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


def get_subject_characteristics(subject_id: int) -> Tuple[str, str]:
    """
    Returns subject class (SCZ or HC) and origin (Basel or Berlin) for a given subject.
    :param subject_id: path to csv file containing participant code.
    :return: two strings, subject class (SCZ or HC) and origin (Basel or Berlin).
    """
    if subject_id < 100:
        if subject_id < 38:
            return 'SCZ', 'Berlin'
        else:
            return 'SCZ', 'Basel'
    else:
        if subject_id < 200:
            return 'HC', 'Berlin'
        else:
            return 'HC', 'Basel'
