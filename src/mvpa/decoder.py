import argparse
from typing import Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib

from src.utils.dataloader import get_subject_ids, load_subject_train_data, average_augment_data
from src.utils.logger import log

from src.mvpa.plot_mvpa import plot_accuracies

matplotlib.use('macOSX')


def init_mvpa():
    """
    Outermost method. Calls function to load, process and decode data (once per subject).
    Arguments are passed from command line.
    :return: None
    """
    title, filename = get_title(args.downsample_factor,
                                args.pseudo_k,
                                args.augment_factor,
                                args.SVM_C,
                                args.SVM_window_width)

    log(title)

    subject_ids = get_subject_ids()

    accuracies = []

    # Loop loads, processes and decodes data one subject at a time
    for subject_id in subject_ids:
        epochs, labels = load_subject_train_data(subject_id,
                                                 downsample_factor=args.downsample_factor)

        if args.pseudo_k > 1:
            epochs, labels = average_augment_data(epochs, labels,
                                                  pseudo_k=args.pseudo_k,
                                                  augment_factor=args.augment_factor)

        log('Decoding subject #{:03d}'.format(subject_id))

        acc = decode_response_over_time(epochs, labels, C=args.SVM_C, window_width=args.SVM_window_width)

        accuracies.append(acc)

    accuracies = np.array(accuracies)

    np.save('results/data/{}.npy'.format(filename), accuracies)

    plot_accuracies(data=accuracies, title=title, savefile=filename, downsample_factor=args.downsample_factor)

    # TODO: test this (also check these subjects are in the data and have correct behaviour files)
    for sid in [21, 24, 40, 42, 106, 116, 206, 208]:
        idx = np.argwhere(subject_ids == sid)[0][0]
        plot_accuracies(data=accuracies[idx],
                        title=f'Subject #{sid}' + title,
                        savefile=f'subj{sid}' + filename,
                        downsample_factor=args.downsample_factor)


def decode_response_over_time(epochs: np.ndarray[float],
                              labels: np.ndarray[int],
                              C: int = 1,
                              window_width: int = 1) -> np.ndarray[float]:
    """
    Decodes response from epoch data. Model: pipeline of scaler and SVM.
    :param epochs: numpy array of epoch data (shape #epochs x #timesteps x #channels).
    :param labels: numpy array of labels (length #epochs).
    :param C: regularisation parameter of SVM.
    :param window_width: width of sliding window (over time) that is fed as input to SVM.
    :return: numpy array of decoding accuracies, one accuracy per time point in the data.
    """
    subject_accuracies = []

    num_epochs = epochs.shape[0]

    # Loop trains SVM for each timestep / window in the epoch data
    for t in range(epochs.shape[1] - window_width + 1):
        pipeline = Pipeline([('scaler', StandardScaler()),
                             ('svc', svm.SVC(kernel='linear', C=C, class_weight='balanced'))])
        scores = cross_val_score(pipeline, np.reshape(epochs[:, t:t + window_width, :], (num_epochs, -1)),
                                 labels, cv=5)
        subject_accuracies.append(np.mean(scores))

    return np.array(subject_accuracies)


def get_title(downsample_factor, pseudo_k, augment_factor, C, window_width) -> Tuple[str, str]:
    """
    Takes MVPA hyperparameters as arguments and returns plot title and file name.
    :param downsample_factor:
    :param pseudo_k:
    :param augment_factor:
    :param C:
    :param window_width:
    :return: title and file name for accuracy plot.
    """
    title = '{} Hz / {}-fold average / {}-fold augment / SVM: C = {}, window = {}'.format(
        int(1000 / downsample_factor),
        pseudo_k,
        augment_factor,
        C,
        window_width)

    filename = 'mvpa_accuracy_{}Hz_av-{}_aug-{}_C-{}_win-{}'.format(int(1000 / downsample_factor),
                                                                    pseudo_k,
                                                                    augment_factor,
                                                                    int(C * 1000),
                                                                    window_width)

    return title, filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments

    parser.add_argument('--downsample_factor',
                        default=5,
                        type=int,
                        required=False,
                        help='Number of timesteps that are collapsed into one when downsampling.')

    parser.add_argument('--pseudo_k',
                        default=1,
                        type=int,
                        required=False,
                        help='Number of trials that are averaged. If 1, no trial averaging is performed.')

    parser.add_argument('--augment_factor',
                        default=1,
                        type=int,
                        required=False,
                        help='Factor with which the data is augmented (only matters when pseudo_k > 1).')

    parser.add_argument('--SVM_C',
                        default=1,
                        type=float,
                        required=False,
                        help='Regularisation parameter of the SVM.')

    parser.add_argument('--SVM_window_width',
                        default=1,
                        type=int,
                        required=False,
                        help='Width of time window that is fed as input to the SVM.')

    args = parser.parse_args()

    init_mvpa()
