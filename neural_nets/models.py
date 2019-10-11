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
        
        # multiply hidden dim of fully-connected linear layer by 2 since LSTM is bidirectional (concat of forward and backward h states)
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
    
    
def fit(model, criterion, optimizer, train_loader, val_loader, classification, epochs):
    counter = 0
    print_every = 10
    clip = 5
    valid_loss_min = np.Inf
    model.train()
    train_losses, train_accs, train_f1_scores = [], [], []
    val_losses, val_accs, val_f1_scores = [], [], []
    for i in range(epochs):
        h = model.init_hidden(batch_size)
        train_losses_epoch, train_accs_epoch, train_f1_epoch = [], [], []
        for inputs, labels in train_loader:
            counter += 1
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([e.data.double() for e in h])
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            if (inputs.shape[0], inputs.shape[1]) == (batch_size, seq_length):
                output, h = model(inputs, h)
                labels = labels.double() if classification == 'binary' else labels.long()
                loss = criterion(output.squeeze(), labels)
                train_losses_epoch.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                train_acc = accuracy(output.squeeze(), labels, task = classification)
                train_f1 = f1(output.squeeze(), labels, task = classification)

                train_accs_epoch.append(train_acc)
                train_f1_epoch.append(train_f1)

            if counter % print_every == 0:
                val_h = model.init_hidden(batch_size)
                val_losses_epoch, val_accs_epoch, val_f1_epoch = [], [], []
                model.eval()
                # no gradient computation for evaluation mode
                with torch.no_grad():
                    for xb_val, yb_val in val_loader:
                        if (xb_val.shape[0], xb_val.shape[1]) == (batch_size, seq_length):
                            val_h = tuple([each.data.double() for each in val_h])
                            inp, lab = xb_val.to(device), yb_val.to(device)
                            out, val_h = model(inp, val_h)

                            lab = lab.double() if classification == 'binary' else lab.long()
                            val_loss = criterion(out.squeeze(), lab)

                            val_acc = accuracy(out, lab, task = classification)
                            val_f1 = f1(out.squeeze(), lab, task= classification)

                            val_accs_epoch.append(val_acc)
                            val_losses_epoch.append(val_loss.item())
                            val_f1_epoch.append(val_f1)

                model.train()
                print("Epoch: {}/{} ".format(i+1, epochs),
                      "Step: {} ".format(counter),
                      "Train Loss: {:.3f} ".format(loss.item()),
                      "Train Acc: {:.3f} ".format(np.mean(train_accs_epoch)),
                      "Train F1: {:.3f} ".format(np.mean(train_f1_epoch)),
                      "Val Loss: {:.3f} ".format(np.mean(val_losses_epoch)),
                      "Val Acc: {:.3f} ".format(np.mean(val_accs_epoch)),
                      "Val F1: {:.3f} ".format(np.mean(val_f1_epoch)))

                if np.mean(val_losses) <= valid_loss_min:
                    torch.save(model.state_dict(), './state_dict.pt')
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                    print()
                    valid_loss_min = np.mean(val_losses)

        train_losses.append(np.mean(train_losses_epoch))
        train_accs.append(np.mean(train_accs_epoch))
        train_f1_scores.append(np.mean(train_f1_epoch))
        val_losses.append(np.mean(val_losses_epoch))
        val_accs.append(np.mean(val_accs_epoch))
        val_f1_scores.append(np.mean(val_f1_epoch))
        
    return train_losses, train_accs, train_f1_scores, val_losses, val_accs, val_f1_scores, model 


def predict(model, criterion, test_loader, classification, batch_size):
    # Loading the best model
    model.load_state_dict(torch.load('./state_dict.pt'))

    test_losses, test_accs, test_f1_scores, preds, true_labels = [], [], [], [], []
    num_correct = 0
    h = model.init_hidden(batch_size)

    model.eval()
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, h = model(inputs, h)
        labels = labels.double() if classification == 'binary' else labels.long()
        test_loss = criterion(output.squeeze(), labels)
        test_losses.append(test_loss.item())
        if classification == 'binary':
            pred = torch.round(output.squeeze())   # Rounds the output to 0/1
            correct_tensor = pred.eq(labels.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)
        else:
            preds.append(torch.argmax(output.squeeze(), dim=1))
            true_labels.append(labels)
            test_acc = accuracy(output.squeeze(), labels, classification)
            test_accs.append(test_acc)
        test_f1_scores.append(f1(output.squeeze(), labels, classification))
    
    test_loss = np.mean(test_losses)
    print("Test loss: {:.3f}".format(test_loss))
    test_acc = num_correct/len(test_loader_rf_bi.dataset) if classification == 'binary' else np.mean(test_accs)
    print("Test accuracy: {:.3f}%".format(test_acc*100))
    test_f1 = np.mean(test_f1_scores)
    print("Test F1-score: {:.3f}%".format(test_f1*100))
    return test_acc, test_f1, test_loss, preds, true_labels