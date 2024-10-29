import numpy as np
import seaborn as sns
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
from utils.dataloader import get_pytorch_dataloader
from utils.logger import log

matplotlib.use('macOSX')
torch.manual_seed(42)

'''
Following PyTorch tutorial (https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

TODOS:
    – FOR NOW: training loss decreases, but validation loss does not (even increases).
        Is the model really only overfitting?
    - Try GRUs instead of LSTMs?
    - For other tips and tricks refer to https://www.ncbi.nlm.nih.gov/books/NBK597502/.

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
        self.linear = nn.Linear(hidden_dim, 1)

        # Pass through tanh to transform output of linear layer to number between -1 and 1
        self.sig = nn.Sigmoid()

    def forward(self, batch):
        batch_size = batch.shape[0]

        # Initialize hidden state (h_0, c_0)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        # Ensure we detach the hidden states between sequences to prevent backprop through time
        # TODO: make sure chatGPT did not gaslight me into doing this.
        h_0 = h_0.detach()
        c_0 = c_0.detach()

        # Pass through LSTM
        h, _ = self.lstm(batch, (h_0, c_0))
        out = self.linear(h)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # HYPERPARAMETERS
    downsample_factor = 1  # when 1 -> memory overload: can I save file in several steps?
    washout = int(1000 / downsample_factor)
    hidden_dim = 64
    num_layers = 1
    learning_rate = 1e-1
    num_epochs = 100

    title = 'LSTM: Accuracy on Validation Set ({} epochs, {} layers, hidden dim = {},  learning rate = {})'.format(
        num_epochs,
        num_layers,
        learning_rate,
        int(hidden_dim))
    filename = '_{:03d}-epochs_{}-layers_{}-hiddim_{}-lr'.format(num_epochs,
                                                                 num_layers,
                                                                 int(learning_rate * 1e4),
                                                                 int(hidden_dim))

    log('Starting LSTM Run')

    print('\nLoading data...')

    dataset = get_pytorch_dataloader(downsample_factor=downsample_factor,
                                     scaled=True)

    # Split lengths (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # test num_workers = 1, 2, 4, ...
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = LSTMPredictor(input_dim=64, hidden_dim=hidden_dim, num_layers=num_layers)
    model = model.to(device)
    model.apply(init_weights)  # probably delete this
    loss_function = nn.BCELoss()
    # TODO: Try momentum. Try other optimizers. Try reducing learning rate once the loss plateaus.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print('\nStarting training...\n')

    epochs_train_loss = np.zeros(num_epochs)
    epochs_validation_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        start = time.time()

        # training loop
        for inputs, labels in train_loader:
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

        # validation loop
        with torch.no_grad():
            model.eval()

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)[:, washout:, :]

                # reshape labels to match output
                labels = labels.unsqueeze(-1).unsqueeze(-1).expand(-1, outputs.shape[1], -1)
                loss = loss_function(outputs, labels)
                epochs_validation_loss[epoch] += loss.item()

            model.train()

        end = time.time()
        print('Epoch [{}/{}]:\n{:>17}: {:8.2f}\n{:>17}: {:8.2f}\n{:>17}: {:8.2f}\n'.format(epoch + 1,
                                                                                           num_epochs,
                                                                                           'Train Loss',
                                                                                           epochs_train_loss[epoch],
                                                                                           'Validation Loss',
                                                                                           epochs_validation_loss[
                                                                                               epoch],
                                                                                           'Elapsed Time',
                                                                                           end - start))

    print('\nSaving model...')

    # timestamp = datetime.datetime.now().strftime("D%Y-%m-%d_T%H-%M-%S")
    torch.save(model.state_dict(), 'models/lstm{}.pth'.format(filename))

    sns.lineplot(y=epochs_train_loss, x=range(1, num_epochs + 1), label='Training Loss')
    sns.lineplot(y=epochs_validation_loss, x=range(1, num_epochs + 1), label='Validation Loss')
    plt.title("Loss over epochs")
    plt.savefig('results/lstm_loss{}.png'.format(filename))

    # for testing
    # model = LSTMPredictor(input_dim=64, hidden_dim=hidden_dim)
    # model.load_state_dict(torch.load('models/...', weights_only=True))

    print('\nEvaluating model...')

    accuracies = torch.Tensor(0)
    with torch.no_grad():
        model.eval()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[:, washout:, :]

            outputs[labels == 0] = 1 - outputs[labels == 0]  # invert scores if label is 0 (to represent accuracy)

            accuracies = torch.cat((accuracies, outputs), dim=0)

        plot_accuracies(data=accuracies.squeeze().numpy(), title=title,
                        savefile='lstm_training_accuracy{}.png'.format(filename),
                        downsample_factor=downsample_factor, washout=washout)

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[:, washout:, :]

            outputs[labels == 0] = 1 - outputs[labels == 0]  # invert scores if label is 0 (to represent accuracy)

            accuracies = torch.cat((accuracies, outputs), dim=0)

        plot_accuracies(data=accuracies.squeeze().numpy(), title=title,
                        savefile='lstm_validation_accuracy{}.png'.format(filename),
                        downsample_factor=downsample_factor, washout=washout)


if __name__ == '__main__':
    init_lstm()
