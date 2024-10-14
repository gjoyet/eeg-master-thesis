import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import rand
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from utils.dataloader import get_subject_ids, load_subject_train_data, average_augment_data

import matplotlib

matplotlib.use('macOSX')
import matplotlib.pyplot as plt


epoch_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/preprocessed/stim_epochs'
behav_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/raw'


'''
Following Lukosevicius et al., Overview of Reservoir Recipes.
'''


class EchoStateNetwork:
    def __init__(self, n_input: int, n_reservoir: int = None, n_output: int = None,
                 sparsity=0.1, spectral_radius=0.9, random_state=None):
        np.random.seed(random_state)

        # Initialize network dimensions
        self.n_input = n_input
        self.n_reservoir = n_input if n_reservoir is None else n_reservoir
        self.n_output = n_input if n_output is None else n_output

        # Input and Reservoir Weight Matrices
        self.W_in = np.random.uniform(-1, 1, (self.n_reservoir, self.n_input))  # Input weights
        self.W_res = rand(self.n_reservoir, self.n_reservoir, density=sparsity).toarray()  # Sparse reservoir weights
        self.W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W_res)))  # Ensure spectral radius < 1

        # Reservoir state
        self.state = np.zeros((self.n_reservoir, 1))

    def _update_reservoir(self, input_signal):
        # Update the reservoir state using tanh activation
        self.state = np.tanh(np.dot(self.W_in, input_signal) + np.dot(self.W_res, self.state))
        return self.state

    def run(self, X, washout=100):
        """
        Train the ESN with input X (time-series data) and target y (labels).
        :param X: Input data of shape (#epochs x #timesteps x #channels)
        :param washout: number of timesteps to ignore at the start.
        :return:
        """
        num_epochs, num_timesteps, num_channels = X.shape
        states = np.zeros(shape=(num_epochs, num_timesteps - washout, self.n_reservoir))

        # Collect the reservoir states for each input
        for i, epoch in enumerate(X):
            self.state = np.zeros((self.n_reservoir, 1))  # Reset reservoir for each new epoch
            for t in range(epoch.shape[0]):
                self._update_reservoir(epoch[t].reshape(-1, 1))
                if t >= washout:
                    states[i, t-washout, :] = self.state.flatten()

        # Scale the reservoir states before training SVM
        scaler = StandardScaler()
        states_scaled = scaler.fit_transform(states)

        return states_scaled
