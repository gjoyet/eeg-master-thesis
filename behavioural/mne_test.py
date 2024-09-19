import os.path
import mne

import matplotlib

matplotlib.use('macOSX')

import matplotlib.pyplot as plt

data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG'


if __name__ == '__main__':
    raw = mne.io.read_raw_brainvision(os.path.join(data_path, 'raw/8/8.vhdr'))
    # raw = mne.read_epochs(os.path.join(data_path, 'preprocessed/stim_epochs/new_eeg_sj2_block1_avg_ref_with_ica_only_muscle_new_minus1000_to_1250ms_stim-epo.fif'))

    print(raw.info)
    print(raw.info.ch_names)

    print(raw.annotations)
    print(set(raw.annotations.description))

    # raw.plot()
    # plt.show()

    # fig = raw.plot_sensors(show_names=True)
    # plt.savefig('results/')
    # plt.show()
