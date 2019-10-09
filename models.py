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
        
def accuracy(out, yb):
    preds = torch.from_numpy(np.array(list(map(lambda proba: 1 if proba > 0.5 else 0, out.detach().numpy())))).double()
    return (preds == yb).float().mean()

#def accuracy(out, yb):
#    preds = torch.round(output.squeeze())
#    return (preds == yb).float().mean()

def get_data_batches(X_train, y_train, X_val, y_val, bs):
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    return (
        DataLoader(train_data, batch_size=bs, shuffle=True),
        DataLoader(val_data, batch_size=bs, shuffle=True),
    )
    
class VanillaLSTM(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, output_size, n_layers, dropout_rate=0.5):
        super(VanillaLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.n_layers = n_layers 
        
        # instantiate layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        #hidden = self.init_hidden(batch_size)
        x = x.double()
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        #out = out.squeeze()[-1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
    
        hidden_state = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        cell_state = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        hidden = (hidden_state, cell_state)
        return hidden