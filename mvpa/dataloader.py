import os.path
import re
from typing import Dict, Tuple, List, Callable

import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from mne.epochs import EpochsFIF


epoched_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/preprocessed/stim_epochs'
behav_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/raw'


def preprocess_train_data(epochs_list: List[np.ndarray[float]], subsampling_rate: int = 1) -> np.ndarray[float]:
    """
    Preprocesses training data.
    Scales channels (each block independently),
    performs PCA retaining 99% of variance (over all epochs).
    :param epochs_list: list of numpy arrays containing epoch data.
             Each array in the list has shape (#epochs x #channels x #timesteps).
    :param subsampling_rate: number of samples that are collapsed into one by averaging.
    :return: numpy array of concatenated processed epoch data (of shape #epochs x #pca_components x #timesteps)
    """
    '''
    For now things like scaling and PCA are calculated on complete data.
    If we want to calculate it on train and apply it on test, we will need to do it differently
    (since for now train/test splits are done inside cross_val_scores).
    -> I could just do my own cross-validation, as the only thing it requires is calculating the
       indices for the folds (something like random_partition(range()) probably exists) and then
       run it once for each fold.
    '''
    proc_epochs = []

    # This loop rescales the data one experiment block at a time. This is because baseline channel activation
    # varies a lot between blocks, but channel activation varies on a much smaller scale inside a block.
    for data in epochs_list:
        num_epochs, num_channels, num_timesteps = data.shape

        # SUBSAMPLE from 1000 Hz to (1000 / subsampling_rate) Hz
        # Ignore first time-step since there are 2251 time-steps, which is not easily divisible.
        data = np.reshape(data[:, :, 1:], shape=(num_epochs, num_channels, num_timesteps // subsampling_rate,
                                                 subsampling_rate))
        data = np.mean(data, axis=-1)

        # SCALE
        scaler = StandardScaler()
        data = reshape_transform_reshape(data=data, func=scaler.fit_transform)

        proc_epochs.append(data)

    proc_epochs = np.concat(proc_epochs, axis=0)

    # PERFORM PCA (on whole data)
    pca = PCA(n_components=0.99, svd_solver='full')  # switching svd_solver to 'auto' might increase performance
    proc_epochs = reshape_transform_reshape(data=proc_epochs, func=pca.fit_transform)

    return proc_epochs


def reshape_transform_reshape(data: np.ndarray[float], func: Callable) -> np.ndarray:
    """
    Transposes and reshapes data such that features are at the last dimension, performs a transformation,
    then reshapes and transposes back.
    :param data: the data to be transformed.
    :param func: the transformation to be applied.
    :return: the transformed data.
    """
    num_epochs, num_channels, num_timesteps = data.shape

    # Transpose and reshape (functions expect features (channels) at last dimension)
    data = np.transpose(data, axes=(0, 2, 1))
    data = np.reshape(data, shape=(num_epochs * num_timesteps, num_channels))

    # Perform transformation
    data = func(data)

    # Transpose back
    data = np.reshape(data, shape=(num_epochs, num_timesteps, -1))
    data = np.transpose(data, axes=(0, 2, 1))

    return data


def load_subject_train_data(subject_id: int) -> Tuple[List[np.ndarray[float]], List[np.ndarray[int]]]:
    """
    Loads and correctly combines epoch data and labels (behavioural outcomes) for one subject.
    Applies baseline and drops NaN labels.
    :param subject_id:
    :return: list of numpy arrays containing epoch data (one per experiment block),
             list of numpy arrays with corresponding labels
    """
    epochs_dict = load_subject_epochs(subject_id)
    results_df = load_subject_labels(subject_id)

    epochs_list = []
    labels_list = []

    for block in epochs_dict.keys():
        epochs = epochs_dict[block]
        epochs = epochs.apply_baseline((-1.000, -0.001))
        labels = (results_df[results_df['run'] == block]['response']).reset_index(drop=True)

        # select labels for which the corresponding epoch was accepted
        selected_labels = np.array(labels.loc[epochs.selection])

        data = epochs.get_data()

        # drop NaN labels and corresponding epochs
        epochs_list.append(data[~np.isnan(selected_labels)])
        labels_list.append(selected_labels[~np.isnan(selected_labels)])

    return epochs_list, labels_list


def load_subject_epochs(subject_id: int) -> Dict[int, EpochsFIF]:
    """
    Loads epoch data for one subject.
    :param subject_id:
    :return: dict, keys are experiment block numbers, values are a mne.Epoch object for the corresponding block.
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
    """
    Loads labels (behavioural outcomes) for one subject.
    :param subject_id:
    :return: a pandas dataframe containing subject response and
             block number ('run') for subsequent join with epoch data.
    """
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

    return np.array(subject_ids)
