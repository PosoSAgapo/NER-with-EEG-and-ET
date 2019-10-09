import numpy as np
import torch.nn.functional as F
import torch

from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    
class VanillaBiLSTM(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, output_size, n_layers, dropout_rate=0.5, task='binary'):
        super(VanillaBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.task = task
        
        # instantiate layers
        biLSTM = False if self.task == 'binary' else True
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_rate, batch_first=True, bidirectional=biLSTM)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # multiply hidden dim of fully-connected linear layer by 2 since LSTM is bidirectional
        hidden_dim = hidden_dim if self.task == 'binary' else hidden_dim * 2
        
        self.fc = nn.Linear(hidden_dim, output_size)
        
        if self.task == 'binary':
            self.pred = nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.double()
        lstm_out, hidden = self.lstm(x, hidden)
        if self.task == 'binary':
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(self.relu(lstm_out))
        out = self.fc(out)
        if self.task == 'binary':
            out = self.pred(out)
            out = out.view(batch_size, -1)
        out = out[:, -1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        layers = self.n_layers if self.task == 'binary' else self.n_layers * 2
        hidden_state = weight.new(layers, batch_size, self.hidden_dim).zero_().to(device)
        cell_state = weight.new(layers, batch_size, self.hidden_dim).zero_().to(device)
        hidden = (hidden_state, cell_state)
        return hidden