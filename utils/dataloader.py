"""
This file is responsible for taking files containing mne.Epochs objects, combining them with
behavioural data (labels) and saving them in numpy format, preparing the data for being used
in classifiers.
"""
import os.path
import re
from typing import Dict, Tuple, List

import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from mne.epochs import EpochsFIF

# Subject IDs are inferred from epoch_data_path. Code in this file expects behavioural_data_path to contain
# data of each subject inside a folder named <subject_id>.
data_root = '/Volumes/Guillaume EEG Project'
epoch_data_path = os.path.join(data_root, 'Berlin_Data/EEG/raw/raw_epochs')
behavioural_data_path = os.path.join(data_root, 'Berlin_Data/EEG/raw')


class CustomNPZDataset(Dataset):
    def __init__(self, file_path):
        # Load the .npz file in 'mmap_mode' for memory-efficient access
        self.data = np.load(file_path, mmap_mode='r')

        # Assume the .npz file contains two arrays: 'inputs' and 'labels'
        self.inputs = self.data['epochs']
        self.labels = self.data['labels']

    def __len__(self):
        return self.inputs.shape[0]  # Return the number of samples (rows)

    def __getitem__(self, idx):
        # Load a single input and label
        input_data = self.inputs[idx]
        label = self.labels[idx]

        # Convert to PyTorch tensors and return
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def get_pytorch_dataset(downsample_factor: int = 1,
                        scaled: bool = True,
                        subject_id: int = None) -> torch.utils.data.Dataset:
    """
    Returns a pytorch dataset with the complete training data.
    Data have applied baseline, dropped NaN labels, and have been downsampled.
    Additionally, the data can be scale scaled.
    :param downsample_factor: number of samples that are collapsed into one by averaging.
    :param scaled: if True, data is scaled.
    :param subject_id: if not None, return a dataset for only that subject.
    :return: pytorch dataloader with training data.
    """
    directory = os.path.join(data_root, 'Data/training_data_{}Hz{}'.format(int(1000 / downsample_factor),
                                                                           '_scaled' if scaled else ''))

    # if file already exists, do nothing
    if not os.path.isdir(directory):
        os.mkdir(directory)
        # else load whole training data and save it in .npz file
        subject_ids = get_subject_ids()
        filenames = []

        for sid in subject_ids:
            epochs, labels = load_subject_train_data(sid, downsample_factor=downsample_factor)

            labels = (labels + 1) / 2  # transform -1 labels to 0 (since we use BCELoss later)

            # SCALE
            # TODO: actually this normalisation might be unnecessary at best, harmful at worst:
            # This normalises features over all timesteps, instead of just over epochs. Might not be
            # what we want. Maybe try BatchNorm or LayerNorm instead (ask Tianlin!).
            if scaled:
                scaler = StandardScaler()
                num_e, num_t, num_c = epochs.shape
                epochs = np.reshape(scaler.fit_transform(np.reshape(epochs, (num_e * num_t, num_c))),
                                    (num_e, num_t, num_c))

            # For smaller files, probably keep save one .npz per subject.
            filename = 'subject{}_training_data_{}Hz{}.npz'.format(sid, int(1000 / downsample_factor),
                                                                   '_scaled' if scaled else '')
            filenames.append(filename)
            np.savez(os.path.join(directory, filename), epochs=epochs, labels=labels)
    else:
        filenames = os.listdir(directory)

    # Define dataset
    if subject_id is None:
        datasets = []
        for fn in filenames:
            datasets.append(CustomNPZDataset(file_path=os.path.join(directory, fn)))

        return torch.utils.data.ConcatDataset(datasets)

    else:
        fn = 'subject{}_training_data_{}Hz{}.npz'.format(subject_id,
                                                         int(1000 / downsample_factor),
                                                         '_scaled' if scaled else '')

        return CustomNPZDataset(file_path=os.path.join(directory, fn))


def neurogpt_prepare_data(downsample_factor: int = 4) -> None:
    savename = 'neurogpt_training_data_{}Hz_RAW'.format(int(1000 // downsample_factor))
    directory = os.path.join(data_root, 'Data', savename)

    if not os.path.isdir(directory):
        os.mkdir(directory)

    subject_ids = get_subject_ids()

    if len(os.listdir(directory)) < len(subject_ids):

        for sid in subject_ids:
            filename = 'subject{}_{}.npz'.format(sid, savename)
            # those subjects are missing channel 'Fz'
            if os.path.isfile(os.path.join(directory, filename)) or sid in [38] + list(range(101, 109)):
                continue

            epochs, labels = neurogpt_load_subject_train_data(sid, downsample_factor)

            labels = (labels + 1) / 2  # transform -1 labels to 0 (since we use BCELoss later)

            # For smaller files, probably keep save one .npz per subject.
            np.savez(os.path.join(directory, filename), epochs=epochs, labels=labels)


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
        # TODO: see if I can fix NaN / inf value issues when interpolating.
        # epochs = epochs.interpolate_bads(reset_bads=False)
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
        epochs = np.reshape(epochs[:, :, 1:], (num_epochs, num_channels, num_timesteps // downsample_factor,
                                               downsample_factor))
        epochs = np.mean(epochs, axis=-1)

    # Transpose to get shape (#epochs x #timesteps x #channels)
    epochs = np.transpose(epochs, axes=(0, 2, 1))

    return epochs, labels


def neurogpt_load_subject_train_data(subject_id: int,
                                     downsample_factor: int = 4) -> Tuple[np.ndarray[float], np.ndarray[int]]:
    """
    Loads and correctly combines epoch data and labels (behavioural outcomes) for one subject.
    Applies baseline, drops NaN labels, downsamples.
    Selects and interpolates channels needed for NeuroGPT.
    :param subject_id:
    :param downsample_factor:
    :return: numpy array containing epoch data (of shape #epochs x #channels x #time),
             numpy array with corresponding labels (of length #epochs).
    """
    epochs_dict = load_subject_epochs(epoch_data_path, subject_id)
    results_df = load_subject_labels(behavioural_data_path, subject_id)

    epochs_list = []
    labels_list = []

    for block in epochs_dict.keys():
        epochs = epochs_dict[block]
        epochs = epochs.apply_baseline((-1.000, -0.001), verbose=False)
        epochs = neurogpt_select_channels(epochs)

        labels = (results_df[results_df['run'] == block]['response']).reset_index(drop=True)

        # select labels for which the corresponding epoch was accepted
        selected_labels = np.array(labels.loc[epochs.selection])

        data = epochs.get_data()[:, :, :2251]

        # drop NaN labels (no response) and corresponding epochs
        epochs_list.append(data[~np.isnan(selected_labels)])
        labels_list.append(selected_labels[~np.isnan(selected_labels)])

    epochs = np.concat(epochs_list, axis=0)
    labels = np.concat(labels_list, axis=0)

    num_epochs, num_channels, num_timesteps = epochs.shape

    # DOWNSAMPLE from 1000 Hz to (1000 / downsample_factor) Hz
    cut = num_timesteps % downsample_factor
    epochs = np.reshape(epochs[:, :, cut:], (num_epochs, num_channels, num_timesteps // downsample_factor,
                                             downsample_factor))
    epochs = np.mean(epochs, axis=-1)

    return epochs, labels


def neurogpt_select_channels(epochs):
    neurogpt_channels = 'Fp1, Fp2, F7, F3, Fz, F4, F8, T1, T3, C3, Cz, C4, T4, T2, T5, P3, Pz, P4, T6, O1, Oz, O2'.split(
        sep=', ')

    # maps from NeuroGPT channels to Berlin Data channels
    channel_dict = {'T3': 'T7',
                    'T4': 'T8',
                    'T5': 'P7',
                    'T6': 'P8',
                    'T1': 'FT9',  # or FT7 ?
                    'T2': 'FT10'}  # or FT8 ?

    selected_channels = [channel_dict.get(ch, ch) for ch in neurogpt_channels]

    epochs = epochs.pick(selected_channels)
    epochs = epochs.reorder_channels(selected_channels)

    return epochs


def load_subject_epochs(path: str, subject_id: int) -> Dict[int, EpochsFIF]:
    """
    Loads epoch data for one subject.
    :param path: path to epoch data.
    :param subject_id:
    :return: dict, keys are experiment block numbers, values are a mne.EpochsFIF object for the corresponding block.
    """
    filenames = os.listdir(path)

    all_epochs = {}

    for fname in filter(lambda k: 'sj{}_'.format(subject_id) in k, filenames):
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

    cols = ['session', 'run', 'response', 'confidence', 'correct', 'choice_rt']
    dfs = []

    # TODO: correct criteria for .csv selection
    for filename in filter(lambda k: ('{}_'.format(subject_id) in k and
                                      'results.csv' in k and
                                      'assr' not in k and
                                      'wrong' not in k),
                           subdirectory_content):
        data = pd.read_csv(os.path.join(path, str(subject_id), filename), usecols=cols)
        if len(data) == 55:
            dfs.append(data)
        else:
            print('Subject #{}: discarding behavioural results file with {} entries.'.format(subject_id,
                                                                                             len(data)))

    combined_df = pd.concat(dfs, ignore_index=True)

    assert len(combined_df == 330), 'Subject #{}: behavioural results have {} entries.'.format(subject_id,
                                                                                               len(combined_df))

    return combined_df


def get_subject_ids() -> np.ndarray:
    """
    Scan through directory to get a list of subject IDs.
    :return: numpy array of subject IDs.
    """
    filenames = os.listdir(epoch_data_path)

    subject_ids = []

    # Regular expression pattern to match "_sj" followed by digits and ending with "_"
    pattern = r"sj(\d+)_"

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


if __name__ == '__main__':
    # IMPORTANT: For now, NeuroGPT data is:
    # WITH applied baseline but WITHOUT scaling (since scaling is done in NeuroGPT code)
    downsample_factor = 1
    neurogpt_prepare_data(downsample_factor)
