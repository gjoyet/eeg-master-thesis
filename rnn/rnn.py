import numpy as np
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from utils.dataloader import get_subject_ids, get_pytorch_dataloader
from utils.logger import log

matplotlib.use('macOSX')
torch.manual_seed(42)

'''
Following PyTorch tutorial (https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

TODOS:
    – FOR NOW: output scores are the same over the whole sequence, and even between sequences...
    – Model not learning: batch/layer normalization? 
    – Save model at the end!
    – Batch it to make it faster?
    - Try GRUs instead of LSTMs?
    - For other tips and tricks refer to https://www.ncbi.nlm.nih.gov/books/NBK597502/.
    
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)  # will need batch_first=True if batched

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim, 1)

        # Pass through tanh to transform output of linear layer to number between -1 and 1
        self.sig = nn.Sigmoid()

    def forward(self, batch):
        batch_size = batch.shape[0]

        # Initialize hidden state (h_0, c_0)
        h_0 = torch.zeros(1, batch_size, self.hidden_dim)  # will need to add a dimension if batched
        c_0 = torch.zeros(1, batch_size, self.hidden_dim)  # will need to add a dimension if batched

        # Ensure we detach the hidden states between sequences to prevent backprop through time
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
    title = 'LSTM Analysis with Default Parameters'
    filename = 'lstm_accuracy'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    downsample_factor = 5  # already have files for 5
    washout = int(1000 / downsample_factor)

    log('Starting LSTM Run')

    subject_ids = get_subject_ids()

    print('\nLoading data...')
    dataloader = get_pytorch_dataloader(subject_ids,
                                        downsample_factor=downsample_factor)

    model = LSTMPredictor(input_dim=64, hidden_dim=64)
    model = model.to(device)
    # model.apply(init_weights)  # probably delete this
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    print('\nStarting training...\n')

    num_epochs = 5
    for epoch in range(num_epochs):
        total_loss = 0
        start = time.time()

        for inputs, labels in dataloader:
            # inputs have shape (batch_size, sequence_length, num_features)
            model.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[:, washout:, :]

            # reshape labels to match output
            labels = labels.unsqueeze(-1).unsqueeze(-1).expand(-1, outputs.shape[1], -1)
            loss = loss_function(outputs, labels)
            print(loss)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        end = time.time()
        print('Epoch [{}/{}]:\n{:>16}: {:8.2f}\n{:>16}: {:8.2f}\n'.format(epoch + 1,
                                                                          num_epochs,
                                                                          'Loss:',
                                                                          total_loss,
                                                                          'Elapsed Time:',
                                                                          end - start))

    print('\nSaving model...')

    # timestamp = datetime.datetime.now().strftime("D%Y-%m-%d_T%H-%M-%S")
    torch.save(model.state_dict(), 'models/rnn.pth')

    # for testing
    # model = LSTMPredictor(input_dim=64, hidden_dim=64)
    # model.load_state_dict(torch.load('models/rnn.pth', weights_only=True))

    print('\nEvaluating model...')

    # very unpretty, but should work
    accuracies = torch.Tensor(0)
    with torch.no_grad():
        model.eval()

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[:, washout:, :]

            outputs[labels == 0] = 1 - outputs[labels == 0]  # invert scores if label is 0 (to represent accuracy)

            accuracies = torch.cat((accuracies, outputs), dim=0)

        accuracies = accuracies.numpy()
        y_plot = np.mean(accuracies, axis=0).flatten()
        x_plot = np.arange(-1000 + washout * downsample_factor, 1250, downsample_factor)
        sns.lineplot(x=x_plot, y=y_plot)
        plt.title('Accuracy After Training')
        plt.show()


if __name__ == '__main__':
    init_lstm()
