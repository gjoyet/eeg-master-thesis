import numpy as np
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from utils.dataloader import get_subject_ids, load_all_train_data
from utils.logger import log


matplotlib.use('macOSX')
torch.manual_seed(42)
epoch_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/preprocessed/stim_epochs'
behavioural_data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/raw'

'''
Following PyTorch tutorial (https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

TODOS:
    - Check that computation of accuracy.
    – Try to understand why the output scores are constant over all timepoints of a sequence.
    – Save model at the end!
    – Batch it to make it faster?
    - Get used to using pytorch 'device' in code for later when I run it on GPUs.
    - Try RNN that produces only one output at the end, train it on different sequence lengths.
'''


class LSTMPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim

        # Trying to embed EEG data could be interesting
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim, 1)

        # Pass through tanh to transform output of linear layer to number between -1 and 1
        self.sig = nn.Sigmoid()

    def forward(self, sequence):
        h, _ = self.lstm(sequence.view(1, len(sequence), -1))
        out = self.linear(h.view(len(sequence), -1))
        score = self.sig(out)
        return score


def init_lstm():
    title = 'LSTM Analysis with Default Parameters'
    filename = 'lstm_accuracy'

    downsample_factor = 5
    washout = 100

    log('Starting LSTM Run')

    subject_ids = get_subject_ids(epoch_data_path)

    accuracies = []

    print('\nLoading data...')
    X, y = load_all_train_data(subject_ids,
                               epoch_data_path=epoch_data_path,
                               behav_data_path=behavioural_data_path,
                               downsample_factor=downsample_factor)

    X = torch.Tensor(X)
    y = torch.Tensor((y + 1) / 2)   # transform -1 labels to 0

    num_samples, num_timesteps, num_channels = X.shape

    model = LSTMPredictor(num_channels, num_channels)
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    # very unpretty, but works
    acc_before_training = []
    with torch.no_grad():
        for sequence, label in zip(X, y):
            scores = model(sequence)[washout:]
            if label == 0:
                scores = 1 - scores  # invert scores if label is 0 (to represent accuracy)
            acc_before_training.append(scores)

        acc_before_training = np.array(acc_before_training)
        y_plot = np.mean(acc_before_training, axis=0).flatten()
        x_plot = np.arange(-500, 1250, 5)
        sns.lineplot(x=x_plot, y=y_plot)
        plt.title('Accuracy Before Training')
        plt.show()

    print('\nStarting training...\n')

    for epoch in range(8):
        total_loss = 0
        start = time.time()

        for sequence, label in zip(X, y):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 3. Run our forward pass.
            scores = model(sequence)[washout:]

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(scores, torch.full_like(scores, label))
            total_loss += loss
            loss.backward()
            optimizer.step()

        end = time.time()
        print('\nLoss at epoch #{:02d}: {:10.6f}\nElapsed Time: {:6.1f}\n'.format(epoch+1, total_loss, end-start))
        
    # very unpretty, but works
    acc_after_training = []
    with torch.no_grad():
        for sequence, label in zip(X, y):
            scores = model(sequence)[washout:]
            if label == 0:
                scores = 1 - scores  # invert scores if label is 0 (to represent accuracy)
            acc_after_training.append(scores)

        acc_after_training = np.array(acc_after_training)
        y_plot = np.mean(acc_after_training, axis=0).flatten()
        x_plot = np.arange(-500, 1250, 5)
        sns.lineplot(x=x_plot, y=y_plot)
        plt.title('Accuracy After Training')
        plt.show()


if __name__ == '__main__':
    init_lstm()
