from typing import Iterable, Tuple

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from mvpa.decoder import plot_accuracies
from utils.dataloader import get_pytorch_dataset
from utils.logger import log

matplotlib.use('macOSX')
torch.manual_seed(42)

'''
Following PyTorch tutorial (https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

TODOS:
    – FOR NOW: training loss decreases, but validation loss does not (even increases).
        Is the model really only overfitting?
    – Try GRUs instead of LSTMs?
    – For other tips and tricks refer to https://www.ncbi.nlm.nih.gov/books/NBK597502/.
    – Try more complex model (inception modules etc.), probably train on raw data.
    – 

    – VALIDATE on separate data. Compare: train separate model for HC/SCZ or one model but validate
        on HC/SCZ separately. Also compare leave x trials out for validation (from random subjects) vs.
        leave whole subjects out.
    – Try regularization (dropout, ...?)
    – Try data averaging/augmentation?
'''


class LSTMPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Trying to embed EEG data could be interesting
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim + input_dim, 1)

        # Pass through tanh to transform output of linear layer to number between -1 and 1
        self.sig = nn.Sigmoid()

    def forward(self, batch):
        batch_size = batch.shape[0]

        # TODO: check what happens if I do not initialize h_0 and c_0 (and hence also do not detach them).
        # Initialize hidden state (h_0, c_0)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        # Ensure we detach the hidden states between sequences to prevent backprop through time
        h_0 = h_0.detach()
        c_0 = c_0.detach()

        # Pass through LSTM
        h, _ = self.lstm(batch, (h_0, c_0))
        hx = torch.cat((h, batch), 2)
        out = self.linear(hx)
        score = self.sig(out)
        return score


class ChronoNet(nn.Module):

    def __init__(self):
        super(ChronoNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            self.MultiscaleConv1D(64, 32),
            # nn.ReLU(),
            self.MultiscaleConv1D(96, 32),
            # nn.ReLU(),
            self.MultiscaleConv1D(96, 32),
            # nn.ReLU(),
        )

        self.gru_layers = nn.ModuleList([
            nn.GRU(96, 32, batch_first=True),
            nn.GRU(32, 32, batch_first=True),
            nn.GRU(64, 32, batch_first=True),
            nn.GRU(96, 32, batch_first=True),
        ])

        self.linear = nn.Linear(32, 1)

        self.sig = nn.Sigmoid()

    def forward(self, batch):
        # Transpose back and forth because CNN modules expect time at last dimension instead of features.
        batch = torch.transpose(batch, 1, 2)
        cnn_out = self.cnn_layers(batch)
        cnn_out = torch.transpose(cnn_out, 1, 2)

        gru_out_0, _ = self.gru_layers[0](cnn_out)
        gru_out_1, _ = self.gru_layers[1](gru_out_0)
        gru_out_2, _ = self.gru_layers[2](torch.cat((gru_out_0, gru_out_1), dim=2))
        gru_out_3, _ = self.gru_layers[3](torch.cat((gru_out_0, gru_out_1, gru_out_2), dim=2))

        # maybe test concatenating with input
        out = self.linear(gru_out_3)
        score = self.sig(out)

        return score

    class MultiscaleConv1D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_sizes: Iterable[int] = (2, 4, 8), stride: int = 1):
            super(ChronoNet.MultiscaleConv1D, self).__init__()
            # iterate the list and create a ModuleList of single Conv1d blocks
            self.kernels = nn.ModuleList()
            for k in kernel_sizes:
                self.kernels.append(nn.Conv1d(in_channels, out_channels, k, stride=stride, padding=k//2 - 1))

        def forward(self, batch):
            # now you can build a single output from the list of convs
            out = [module(batch) for module in self.kernels]
            # concatenate at dim=1 since in convolutions features are at dim=1
            return torch.cat(out, dim=1)


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


def get_plot_title(simple, num_epochs, learning_rate, weight_decay,
                   num_layers=1, hidden_dim=64, downsampling=True, subject_id=None) -> Tuple[str, str]:
    """
    Takes hyperparameters as arguments and returns plot title and file name.
    :param simple:
    :param num_epochs:
    :param learning_rate:
    :param weight_decay:
    :param num_layers:
    :param hidden_dim:
    :param downsampling:
    :param subject_id:
    :return:
    """
    if simple:
        title = 'LSTM{}: ({} epochs, {} layers, hiddim = {}, lr = {}, wd = {})'.format(
            f' on subject #{subject_id}' if subject_id is not None else '',
            num_epochs,
            num_layers,
            int(hidden_dim),
            learning_rate,
            weight_decay)

        filename = 'lstm_{}{:03d}-epochs{}'.format(
            f'sj{subject_id}_' if subject_id is not None else '',
            num_epochs,
            '_1000Hz' if not downsampling else '')

    else:
        title = 'ChronoNet{}: ({} epochs, lr = {}, wd = {})'.format(
            f' on subject #{subject_id}' if subject_id is not None else '',
            num_epochs,
            learning_rate,
            weight_decay)

        filename = 'chrononet_{}{:03d}-epochs{}'.format(
            f'sj{subject_id}_' if subject_id is not None else '',
            num_epochs,
            '_1000Hz' if not downsampling else '')

    return title, filename


def init_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # HYPERPARAMETERS
    downsample_factor = 5
    washout_factor = 1000 / 2250
    learning_rate = 1e-4  # 1e-4 for simple model, ??? for ChronoNet
    weight_decay = 1
    num_epochs = 10
    simple = False
    hidden_dim = 64  # only relevant when simple = True
    num_layers = 1   # only relevant when simple = True

    subject_id = 111  # if None, train on all subjects
    compute_accuracies = True  # False makes script quicker for testing
    title, filename = get_plot_title(simple=simple, num_epochs=num_epochs,
                                     learning_rate=learning_rate, weight_decay=weight_decay,
                                     num_layers=num_layers, hidden_dim=hidden_dim,
                                     subject_id=subject_id)

    log('Starting LSTM Run')

    print('\nLoading data...')

    dataset = get_pytorch_dataset(downsample_factor=downsample_factor,
                                  scaled=True, subject_id=subject_id)

    # Split lengths (e.g., 80% train, 20% test)
    split = 0.8 if subject_id is None else 0.65
    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    # TODO: see what happens when I change batch size.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # test num_workers = 1, 2, 4, ...
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Define model
    if simple:
        model = LSTMPredictor(input_dim=64, hidden_dim=hidden_dim, num_layers=num_layers)
    else:
        model = ChronoNet()

    model = model.to(device)
    model.apply(init_weights)  # probably delete this
    loss_function = nn.BCELoss()
    # TODO: Try momentum. Try other optimizers. Try reducing learning rate once the loss plateaus.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print('\nStarting training...\n')

    epochs_train_loss = np.zeros(num_epochs + 1)
    epochs_validation_loss = np.zeros(num_epochs + 1)

    with torch.no_grad():
        model.eval()

        for i, (inputs, labels) in enumerate(train_loader):
            # inputs have shape (batch_size, sequence_length, num_features)
            inputs, labels = inputs.to(device), labels.to(device)
            if i == 0:
                outputs = model(inputs)
                washout = int(outputs.shape[1] * washout_factor)

            outputs = model(inputs)[:, washout:, :]

            # reshape labels to match output
            labels = labels.unsqueeze(-1).unsqueeze(-1).expand(-1, outputs.shape[1], -1)
            loss = loss_function(outputs, labels)
            epochs_train_loss[0] += loss.item()

        epochs_train_loss[0] /= i + 1

        for i, (inputs, labels) in enumerate(test_loader):
            # inputs have shape (batch_size, sequence_length, num_features)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[:, washout:, :]

            # reshape labels to match output
            labels = labels.unsqueeze(-1).unsqueeze(-1).expand(-1, outputs.shape[1], -1)
            loss = loss_function(outputs, labels)
            epochs_validation_loss[0] += loss.item()

        epochs_validation_loss[0] /= i + 1

        model.train()

    print('Initial loss: {}\n'.format(epochs_train_loss[0]))

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        # training loop
        for i, (inputs, labels) in enumerate(train_loader):
            # inputs have shape (batch_size, sequence_length, num_features)
            model.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[:, washout:, :]

            # reshape labels to match output
            labels = labels.unsqueeze(-1).unsqueeze(-1).expand(-1, outputs.shape[1], -1)
            loss = loss_function(outputs, labels)
            epochs_train_loss[epoch] += loss.item()

            loss.backward()
            optimizer.step()

        epochs_train_loss[epoch] /= i + 1

        # validation loop
        with torch.no_grad():
            model.eval()

            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)[:, washout:, :]

                # reshape labels to match output
                labels = labels.unsqueeze(-1).unsqueeze(-1).expand(-1, outputs.shape[1], -1)
                loss = loss_function(outputs, labels)
                epochs_validation_loss[epoch] += loss.item()

            epochs_validation_loss[epoch] /= i + 1
            model.train()

        end = time.time()
        print('Epoch [{}/{}]:\n{:>17}: {:>8.7f}\n{:>17}: {:>8.7f}\n{:>17}: {:>8.2f}\n'.format(epoch,
                                                                                              num_epochs,
                                                                                              'Train Loss',
                                                                                              epochs_train_loss[epoch],
                                                                                              'Validation Loss',
                                                                                              epochs_validation_loss[
                                                                                                  epoch],
                                                                                              'Elapsed Time',
                                                                                              end - start))

    print('\nMinimum validation loss: {:>8.7f}'.format(np.min(epochs_validation_loss)))
    print('\nSaving model...')

    # timestamp = datetime.datetime.now().strftime("D%Y-%m-%d_T%H-%M-%S")
    torch.save(model.state_dict(), 'models/{}.pth'.format(filename))

    sns.lineplot(y=epochs_train_loss, x=range(num_epochs + 1), label='Training Loss')
    sns.lineplot(y=epochs_validation_loss, x=range(num_epochs + 1), label='Validation Loss')
    plt.title("Loss over epochs")
    plt.savefig('results/{}_loss.png'.format(filename))
    plt.show()

    # for testing
    # model = LSTMPredictor(input_dim=64, hidden_dim=hidden_dim)
    # model.load_state_dict(torch.load('models/...', weights_only=True))

    print('\nEvaluating model...')

    if not compute_accuracies:
        return

    with torch.no_grad():
        model.eval()

        trainset_scores = torch.empty(0)
        trainset_accuracies = torch.empty(0)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[:, washout:, :]

            predictions = outputs >= 0.5
            accuracy = predictions == labels.unsqueeze(-1).unsqueeze(-1).expand(-1, outputs.shape[1], -1)

            outputs[labels == 0] = 1 - outputs[labels == 0]  # invert scores if label is 0 (to represent accuracy)

            trainset_accuracies = torch.cat((trainset_accuracies, accuracy), dim=0)
            trainset_scores = torch.cat((trainset_scores, outputs), dim=0)

        testset_scores = torch.empty(0)
        testset_accuracies = torch.empty(0)
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[:, washout:, :]

            predictions = outputs >= 0.5
            accuracy = predictions == labels.unsqueeze(-1).unsqueeze(-1).expand(-1, outputs.shape[1], -1)

            outputs[labels == 0] = 1 - outputs[labels == 0]  # invert scores if label is 0 (to represent accuracy)

            testset_accuracies = torch.cat((testset_accuracies, accuracy), dim=0)
            testset_scores = torch.cat((testset_scores, outputs), dim=0)

        plot_accuracies(data=trainset_scores.squeeze().numpy(),
                        title='Training Scores {}'.format(title),
                        savefile='{}_train_score'.format(filename),
                        downsample_factor=downsample_factor, washout=washout)

        plot_accuracies(data=testset_scores.squeeze().numpy(),
                        title='Validation Scores {}'.format(title),
                        savefile='{}_test_score'.format(filename),
                        downsample_factor=downsample_factor, washout=washout)

        plot_accuracies(data=trainset_accuracies.squeeze().numpy(),
                        title='Training Accuracy {}'.format(title),
                        savefile='{}_train_acc'.format(filename),
                        downsample_factor=downsample_factor, washout=washout)

        plot_accuracies(data=testset_accuracies.squeeze().numpy(),
                        title='Validation Accuracy {}'.format(title),
                        savefile='{}_test_acc'.format(filename),
                        downsample_factor=downsample_factor, washout=washout)


if __name__ == '__main__':
    init_lstm()
