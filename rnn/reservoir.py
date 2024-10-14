import argparse

import numpy as np
from scipy.sparse import rand

from mvpa.decoder import decode_response_over_time, plot_accuracies
from utils.dataloader import get_subject_ids, load_subject_train_data
from utils.logger import log


epoch_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/preprocessed/stim_epochs'
behav_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/raw'


'''
Following Lukosevicius et al., Overview of Reservoir Recipes.

TODOS:
    - NEXT STEP: run and test this.
'''


def init_esn():
    """
    Outermost method. Calls function to load, process and decode data (once per subject).
    Arguments are passed from command line.
    :return: None
    """
    title = 'ESN Analysis with Default Parameters'
    filename = 'esn_accuracy'
    downsample_factor = 5

    log('Starting ESN Run')

    subject_ids = get_subject_ids(epoch_data_path)

    accuracies = []

    # Loop loads, processes and decodes data one subject at a time
    for subject_id in subject_ids:
        epochs, labels = load_subject_train_data(subject_id,
                                                 epoch_data_path=epoch_data_path,
                                                 behav_data_path=behav_data_path,
                                                 downsample_factor=downsample_factor)

        esn = EchoStateNetwork(input_dim=epochs.shape[-1])
        epochs = esn.run(epochs)

        log('Decoding subject #{:03d}'.format(subject_id))

        acc = decode_response_over_time(epochs, labels, C=1, window_width=1)

        accuracies.append(acc)

    accuracies = np.array(accuracies)

    np.save('results/data/{}.npy'.format(filename), accuracies)

    # TODO: correct plot x-axis labels to take washout into account
    plot_accuracies(data=accuracies, title=title, savefile=filename, downsample_factor=downsample_factor)


class EchoStateNetwork:
    def __init__(self, input_dim: int, reservoir_dim: int = None, output_dim: int = None,
                 sparsity=0.1, spectral_radius=0.9, random_state=None):
        np.random.seed(random_state)

        # Initialize network dimensions
        self.input_dim = input_dim
        self.reservoir_dim = input_dim if reservoir_dim is None else reservoir_dim
        self.output_dim = input_dim if output_dim is None else output_dim

        # Input and Reservoir Weight Matrices
        self.W_in = np.random.uniform(-1, 1, (self.reservoir_dim, self.input_dim))  # Input weights
        self.W_res = rand(self.reservoir_dim, self.reservoir_dim, density=sparsity).toarray()  # Sparse reservoir weights
        self.W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W_res)))  # Ensure spectral radius < 1

        # Reservoir state
        self.state = np.zeros((self.reservoir_dim, 1))

    def _update_reservoir(self, input_signal):
        # Update the reservoir state using tanh activation
        self.state = np.tanh(np.dot(self.W_in, input_signal) + np.dot(self.W_res, self.state))
        return self.state

    def run(self, X, washout=100) -> np.ndarray[float]:
        """
        Train the ESN with input X (time-series data) and target y (labels).
        :param X: Input data of shape (#epochs x #timesteps x #channels)
        :param washout: number of timesteps to ignore at the start.
        :return:
        """
        num_epochs, num_timesteps, num_channels = X.shape
        states = np.zeros(shape=(num_epochs, num_timesteps - washout, self.reservoir_dim))

        # Collect the reservoir states for each input
        for i, epoch in enumerate(X):
            self.state = np.zeros((self.reservoir_dim, 1))  # Reset reservoir for each new epoch
            for t in range(epoch.shape[0]):
                self._update_reservoir(epoch[t].reshape(-1, 1))
                if t >= washout:
                    states[i, t-washout, :] = self.state.flatten()

        return states


if __name__ == '__main__':
    init_esn()
