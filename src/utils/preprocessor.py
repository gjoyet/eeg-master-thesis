"""
This file is responsible for producing files containing mne.Epochs objects from raw EEG files.
"""
import os
import re
from typing import List

import numpy as np
import pandas as pd
import mne
import shutil

from src.utils import logger

data_root = '/Volumes/Guillaume EEG Project'
raw_data_path = os.path.join(data_root, 'Berlin_Data/EEG/raw')
behavioural_data_path = os.path.join(data_root, 'Berlin_Data/EEG/raw')
save_data_path = os.path.join(data_root, 'Berlin_Data/Angeline/EEG_prepared')


def init() -> None:
    subject_ids = get_subject_ids()

    if not os.path.isdir(save_data_path):
        os.mkdir(save_data_path)

    for sid in subject_ids:
        # subjects with too little/many blocks (except 124, 22 there the block has only weird stimuli value counts)
        if os.path.isdir(os.path.join(save_data_path, '{}'.format(sid))) or \
                sid in [20, 16, 127, 10, 2, 29, 137, 124, 22]:
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

    save_as_blocks_and_rename(raws, subject_id)


def save_as_blocks_and_rename(raws: List[mne.io.Raw], subject_id: int) -> None:
    def adjust_events(raw, events):
        """ Keep only events within the range of the raw data """
        start_sample = int(raw.first_samp)  # First sample of cropped data
        stop_sample = int(raw.last_samp)  # Last sample of cropped data
        valid_events = events[(events[:, 0] >= start_sample) & (events[:, 0] <= stop_sample)]

        # Shift event onset times relative to new start
        valid_events[:, 0] -= start_sample
        return valid_events

    raw_segments = []
    for raw in raws:
        events, event_id = mne.events_from_annotations(raw, verbose='WARNING')

        # Find the onset times of the relevant annotations
        split_onsets = [annot['onset']-1 for annot in raw.annotations if annot['description'] == 'Stimulus/S105']
        split_offsets = []

        trial_times = np.array([annot['onset'] for annot in raw.annotations if annot['description'] == 'Stimulus/S 64'])
        for t in split_onsets[1:] + [raw.times[-1]]:
            # cut at the last trial of the block + 10 seconds
            if len(trial_times[trial_times < t]) == 0:
                split_onsets.remove(t)
                continue
            split_offsets.append(np.min([np.max(trial_times[trial_times < t]) + 20,
                                         raw.times[-1]]))

        # Iterate over consecutive pairs of split points to create new raw segments
        for start, end in zip(split_onsets, split_offsets):
            if start >= end:
                print('Discarding empty segment.')
                continue
            raw_segment = raw.copy().crop(tmin=start, tmax=end, include_tmax=False)
            # print(pd.Series(raw_segment.annotations.description).value_counts())
            num_trials = pd.Series(raw_segment.annotations.description).value_counts()['Stimulus/S 64']

            if num_trials == 55:
                adjusted_events = adjust_events(raw_segment, events)
                adjusted_annotations = mne.annotations_from_events(adjusted_events, raw_segment.info['sfreq'],
                                                                   event_desc=dict((v, k) for k, v in event_id.items()))
                raw_segment.set_annotations(adjusted_annotations)

                raw_segments.append(raw_segment)
            else:
                print('Discarding raw segment with {} trials.'.format(num_trials))

    behavioural_filenames = get_behavioural_filenames(behavioural_data_path, subject_id)

    assert len(raw_segments) == len(behavioural_filenames), \
        'Subject #{} has {} raw segments but {} behavioural files'.format(subject_id,
                                                                          len(raw_segments),
                                                                          len(behavioural_filenames))

    os.mkdir(os.path.join(save_data_path,
                          '{}'.format(subject_id)))
    os.mkdir(os.path.join(save_data_path,
                          '{}'.format(subject_id),
                          '2afc'))
    for block_n, seg in enumerate(raw_segments):
        mne.export.export_raw(os.path.join(save_data_path,
                                           '{}'.format(subject_id),
                                           '{}_2afc_run{}.vhdr'.format(subject_id, block_n + 1)),
                              seg,
                              fmt='brainvision')
    for fn in behavioural_filenames:
        shutil.copyfile(fn, os.path.join(save_data_path,
                                         '{}'.format(subject_id),
                                         '2afc',
                                         os.path.basename(fn)))


def get_behavioural_filenames(path: str, subject_id: int) -> List[str]:
    subdirectory_content = os.listdir(os.path.join(path, str(subject_id)))
    filenames = [os.path.join(path, str(subject_id), el) for el in filter(lambda k: ('{}_'.format(subject_id) in k and
                                                                                     'results.csv' in k and
                                                                                     'assr' not in k and
                                                                                     'wrong' not in k),
                                                                          subdirectory_content)]

    dfs = []
    # TODO: correct criteria for .csv selection
    for filename in filenames:
        data = pd.read_csv(os.path.join(path, str(subject_id), filename))
        if len(data) == 55:
            dfs.append(data)
        else:
            filenames.remove(filename)
            print('Subject #{}: discarding behavioural results file with {} entries.'.format(subject_id,
                                                                                             len(data)))

    return filenames


# @obsolete
# def save_epochs_from_raw(raws: List[mne.io.Raw], subject_id: int) -> None:
#     raw_segments = []
#     for raw in raws:
#         # Find the onset times of the relevant annotations
#         split_onsets = [annot['onset'] for annot in raw.annotations if annot['description'] == 'Stimulus/S105']
#
#         # Sort the onsets and add the start and end times of the raw object for splitting
#         split_onsets = split_onsets + [raw.times[-1]]
#
#         # Iterate over consecutive pairs of split points to create new raw segments
#         for start, end in zip(split_onsets[:-1], split_onsets[1:]):
#             raw_segment = raw.copy().crop(tmin=start, tmax=end, include_tmax=False)
#             # print(pd.Series(raw_segment.annotations.description).value_counts())
#             num_trials = pd.Series(raw_segment.annotations.description).value_counts()['Stimulus/S 64']
#             if num_trials == 55:
#                 raw_segments.append(raw_segment)
#             else:
#                 print('Discarding raw segment with {} trials.'.format(num_trials))
#
#     assert len(raw_segments) == 6, 'Subject #{} has {} blocks'.format(subject_id, len(raw_segments))
#
#     # Define the epochs
#     tmin = -1.0  # Start of each epoch
#     tmax = 2.5  # End of each epoch
#
#     for block_n, seg in enumerate(raw_segments):
#         events, event_id = mne.events_from_annotations(seg, event_id={'Stimulus/S 64': 1})
#         epochs = (mne.Epochs(seg, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True))
#         epochs.save(os.path.join(save_data_path,
#                                  'sj{}_block{}_-1000ms_to_2500ms_not_preprocessed-epo.fif'.format(subject_id,
#                                                                                                   block_n + 1)))


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
