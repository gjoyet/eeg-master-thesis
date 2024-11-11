import os

# Subject IDs are inferred from epoch_data_path. Code in this file expects behavioural_data_path to contain
# data of each subject inside a folder named <subject_id>.
data_root = '/Volumes/Guillaume EEG Project'
epoch_data_path = os.path.join(data_root, 'Berlin_Data/EEG/preprocessed/stim_epochs_incl_response')
raw_data_path = os.path.join(data_root, 'Berlin_Data/EEG/raw')


def preprocess_subject():
    pass


def epochs_from_raw():
    pass


if __name__ == '__main__':
    pass
