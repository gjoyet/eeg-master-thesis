import numpy as np

from src.mvpa.decoder import decode_response_over_time, plot_accuracies
from src.utils.dataloader import get_subject_ids, load_subject_train_data
from src.utils.logger import log


'''
Following Lukosevicius et al., Overview of Reservoir Recipes.

TODOS:
    - NEXT STEPS: 
        + reduce redundancies between mvpa/decoder.py and this file.
            
'''


class EchoStateNetwork:
    def __init__(self, input_dim: int, reservoir_dim: int = None, output_dim: int = None,
                 density=0.2, spectral_radius=0.9, random_state=None):
        np.random.seed(random_state)

        # Initialize network dimensions
        self.input_dim = input_dim
        self.reservoir_dim = input_dim if reservoir_dim is None else reservoir_dim
        self.output_dim = input_dim if output_dim is None else output_dim

        # Input and Reservoir Weight Matrices
        self.W_in = np.random.uniform(-1, 1, (self.reservoir_dim, self.input_dim))  # input weights
        self.W_res = np.random.uniform(-1, 1, (self.reservoir_dim, self.reservoir_dim))  # reservoir weights
        self.W_res *= np.random.choice([0, 1], size=self.W_res.shape, p=[1 - density, density])  # make them sparse
        self.W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W_res)))  # ensure spectral radius < 1

        # Reservoir state
        self.state = np.zeros((self.reservoir_dim, 1))

    def _update_reservoir(self, input_signal):
        # Update the reservoir state using tanh activation
        self.state = np.tanh(np.dot(self.W_in, input_signal) + np.dot(self.W_res, self.state))
        return self.state

    def run(self, epochs, washout=100) -> np.ndarray[float]:
        """
        Runs epochs through the ESN, returns the series of states.
        :param epochs: Input data of shape (#epochs x #timesteps x #channels)
        :param washout: number of timesteps to ignore at the start.
        :return: numpy array of shape (#epochs x #timesteps #state_dimension)
        """
        num_epochs, num_timesteps, num_channels = epochs.shape
        states = np.zeros(shape=(num_epochs, num_timesteps - washout, self.reservoir_dim))

        # Collect the reservoir states for each input
        for i, epoch in enumerate(epochs):
            self.state = np.zeros((self.reservoir_dim, 1))  # Reset reservoir for each new epoch
            for t, sample in enumerate(epoch):
                self._update_reservoir(sample.reshape(-1, 1))
                if t >= washout:
                    states[i, t-washout, :] = self.state.flatten()

        return states


def init_esn():
    """
    Outermost method. Calls function to load, process and decode data (once per subject).
    Arguments are passed from command line.
    :return: None
    """
    title = 'ESN Analysis with Default Parameters'
    filename = 'esn_accuracy'

    downsample_factor = 5
    washout = int(500 / downsample_factor)

    log('Starting ESN Run')

    subject_ids = get_subject_ids()

    accuracies = []

    # Loop loads, processes and decodes data one subject at a time
    for subject_id in subject_ids:
        epochs, labels = load_subject_train_data(subject_id,
                                                 downsample_factor=downsample_factor)

        esn = EchoStateNetwork(input_dim=epochs.shape[-1])
        epochs = esn.run(epochs, washout=washout)

        log('Decoding subject #{:03d}'.format(subject_id))

        acc = decode_response_over_time(epochs, labels, C=1, window_width=1)

        accuracies.append(acc)

    accuracies = np.array(accuracies)

    np.save('results/data/{}.npy'.format(filename), accuracies)

    plot_accuracies(data=accuracies, title=title, savefile=filename,
                    downsample_factor=downsample_factor, washout=washout)


if __name__ == '__main__':
    init_esn()
