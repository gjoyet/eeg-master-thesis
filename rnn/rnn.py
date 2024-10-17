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
    – FOR NOW: output scores are the same over the whole sequence, and even between sequences...
    – Model not learning: batch/layer normalization? 
    – Save model at the end!
    – Batch it to make it faster?
    
    - How can I train on data while not holding the whole data in memory all the time?
    - Check that computation of accuracy.
    
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
        self.lstm = nn.LSTM(input_dim, hidden_dim)  # will need batch_first=True if batched

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim, 1)

        # Pass through tanh to transform output of linear layer to number between -1 and 1
        self.sig = nn.Sigmoid()

    def forward(self, sequence):
        # Initialize hidden state (h_0, c_0)
        h_0 = torch.zeros(1, self.hidden_dim)  # will need to add a dimension if batched
        c_0 = torch.zeros(1, self.hidden_dim)  # will need to add a dimension if batched

        # Ensure we detach the hidden states between sequences to prevent backprop through time
        h_0 = h_0.detach()
        c_0 = c_0.detach()

        # Pass through LSTM
        h, _ = self.lstm(sequence, (h_0, c_0))
        out = self.linear(h.view(len(sequence), -1))
        score = self.sig(out)
        return score


# probably delete this
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


def init_lstm():
    title = 'LSTM Analysis with Default Parameters'
    filename = 'lstm_accuracy'

    downsample_factor = 50
    washout = int(1000 / downsample_factor)

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

    # Make data smaller for testing
    # TODO: delete afterwards
    # np.random.seed(seed=42)
    # idxs = np.random.choice(len(y), size=len(y)//50)
    # X = X[idxs]
    # y = y[idxs]

    num_samples, num_timesteps, num_channels = X.shape

    model = LSTMPredictor(num_channels, num_channels)
    # model.apply(init_weights)  # probably delete this
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    print('\nStarting training. Number of sequences: {}\n'.format(X.shape[0]))

    for epoch in range(5):
        total_loss = 0
        start = time.time()

        for sequence, label in zip(X, y):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 3. Run our forward pass.
            scores = model(sequence)[washout:]

            # Step 4. Compute the loss, gradients, and update the parameters
            loss = loss_function(scores, torch.full_like(scores, label))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        end = time.time()
        print('\nLoss at epoch #{:02d}: {:8.2f}\nElapsed Time: {:6.1f}\n'.format(epoch+1, total_loss, end-start))
        
    # very unpretty, but should work
    acc_after_training = []
    with torch.no_grad():
        for sequence, label in zip(X, y):
            scores = model(sequence)[washout:]
            if label == 0:
                scores = 1 - scores  # invert scores if label is 0 (to represent accuracy)
            acc_after_training.append(scores)

        acc_after_training = np.array(acc_after_training)
        y_plot = np.mean(acc_after_training, axis=0).flatten()
        x_plot = np.arange(-1000 + washout * downsample_factor, 1250, downsample_factor)
        sns.lineplot(x=x_plot, y=y_plot)
        plt.title('Accuracy After Training')
        plt.show()


if __name__ == '__main__':
    init_lstm()
