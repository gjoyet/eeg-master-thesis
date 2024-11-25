"""
This file is responsible for producing files containing mne.Epochs objects from raw EEG files.
"""
import os
import re
from typing import List

import pandas as pd
import mne

from utils import logger

data_root = '/Volumes/Guillaume EEG Project'
raw_data_path = os.path.join(data_root, 'Berlin_Data/EEG/raw')
save_data_path = os.path.join(data_root, 'Berlin_Data/EEG/raw/raw_epochs')


def init() -> None:
    subject_ids = get_subject_ids()

    if not os.path.isdir(save_data_path):
        os.mkdir(save_data_path)

    for sid in subject_ids:
        # subjects with too little/many blocks (except 124, 22 there the block has only weird stimuli value counts)
        if sid in [20, 16, 127, 10, 2, 124, 22]:
            continue
        preprocess_subject(sid)


def preprocess_subject(subject_id: int) -> None:
    logger.log('Preprocessing Subject #{}'.format(subject_id))
    dir_content = os.listdir(os.path.join(raw_data_path, str(subject_id)))
    pattern = r'^(proband_)?\d+(_scz)?(_2afc)?(_(run)?\d+)?(_take2)?\.vhdr$'
    raw_files = [fn for fn in dir_content if re.match(pattern, fn)]
    raw_files.sort()

    raws = []
    for fn in raw_files:
        raw = mne.io.read_raw_brainvision(os.path.join(raw_data_path, str(subject_id), fn))
        raws.append(raw)

    save_epochs_from_raw(raws, subject_id)


def save_epochs_from_raw(raws: List[mne.io.Raw], subject_id: int) -> None:
    raw_segments = []
    for raw in raws:
        # Find the onset times of the relevant annotations
        split_onsets = [annot['onset'] for annot in raw.annotations if annot['description'] == 'Stimulus/S105']

        # Sort the onsets and add the start and end times of the raw object for splitting
        split_onsets = split_onsets + [raw.times[-1]]

        # Iterate over consecutive pairs of split points to create new raw segments
        for start, end in zip(split_onsets[:-1], split_onsets[1:]):
            raw_segment = raw.copy().crop(tmin=start, tmax=end, include_tmax=False)
            # print(pd.Series(raw_segment.annotations.description).value_counts())
            num_trials = pd.Series(raw_segment.annotations.description).value_counts()['Stimulus/S 64']
            if num_trials == 55:
                raw_segments.append(raw_segment)
            else:
                print('Discarding raw segment with {} trials.'.format(num_trials))

    assert len(raw_segments) == 6, 'Subject #{} has {} blocks'.format(subject_id, len(raw_segments))

    # Define the epochs
    tmin = -1.0  # Start of each epoch
    tmax = 2.5  # End of each epoch

    for block_n, seg in enumerate(raw_segments):
        events, event_id = mne.events_from_annotations(seg, event_id={'Stimulus/S 64': 1})
        epochs = (mne.Epochs(seg, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True))
        epochs.save(os.path.join(save_data_path,
                                 'sj{}_block{}_-1000ms_to_2500ms_not_preprocessed-epo.fif'.format(subject_id,
                                                                                                  block_n + 1)))


def get_subject_ids() -> List[int]:
    dir_content = os.listdir(raw_data_path)
    subject_ids = [int(sid) for sid in dir_content if sid.isdigit()]

    try:
        subject_ids.remove(9)
        subject_ids.remove(18)
        subject_ids.remove(34)
        subject_ids.remove(117)
    except ValueError:
        pass

    return subject_ids


if __name__ == '__main__':
    init()
