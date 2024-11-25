import os.path
import mne
import pandas as pd

import matplotlib

matplotlib.use('macOSX')

import matplotlib.pyplot as plt

data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG'


if __name__ == '__main__':
    raw = mne.io.read_raw_brainvision(os.path.join(data_path, 'raw/38/101_2afc_run1.vhdr'))
    raw_epochs = mne.read_epochs(os.path.join(data_path, 'preprocessed/stim_epochs/new_eeg_sj38_block1_avg_ref_with_ica_only_muscle_new_minus1000_to_1250ms_stim-epo.fif'))

    print(raw.info)
    print(raw.info.ch_names)
    print(raw_epochs.info.ch_names)

    # TODO: How can the epochs have more channels than raw?????????????

    neurogpt_ch = 'Fp1, Fp2, F7, F3, Fz, F4, F8, T1, T3, C3, Cz, C4, T4, T2, T5, P3, Pz, P4, T6, O1, Oz, O2'.split(sep=', ')
    print(neurogpt_ch)

    count = 0
    common_ch = []
    missing_ch = []

    for ch in neurogpt_ch:
        if ch in raw.info.ch_names:
            common_ch.append(ch)
            count += 1
        else:
            missing_ch.append(ch)

    print('\n\n# channels in common with NeuroGPT: {} out of {}'.format(count, len(neurogpt_ch)))
    print('Common channels: {}'.format(common_ch))
    print('Missing channels: {}\n\n'.format(missing_ch))

    print(raw.annotations)
    print(set(raw.annotations.description))

    raw.plot()
    plt.show()

    fig = raw.plot_sensors(show_names=True)
    plt.show()

    mne.viz.plot_montage(mne.channels.read_custom_montage('/Users/joyet/Downloads/CACS-64_NO_REF.bvef'))
    plt.show()

    for mon in mne.channels.get_builtin_montages():
        mne.viz.plot_montage(mne.channels.make_standard_montage(mon))
        plt.show()
