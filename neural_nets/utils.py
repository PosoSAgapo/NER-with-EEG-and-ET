import matplotlib.pyplot as plt
import numpy as np
import torch.functional as F
import torch 
import os

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

def loss_func(classification:str, weight=None, mask_index=-100): 
    return nn.BCELoss() if classification == 'binary' else nn.CrossEntropyLoss(weight, ignore_index=mask_index)

def get_optim(model, lr:float):
    return torch.optim.Adam(model.parameters(), lr=lr)

def accuracy(out, yb, task:str):
    if task == 'binary':
        preds = torch.from_numpy(np.array(list(map(lambda proba: 1 if proba > 0.5 else 0, out.detach().numpy())))).double()
    else:
        preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def f1(out, yb, task:str):
    if task == 'binary':
        preds = torch.from_numpy(np.array(list(map(lambda proba: 1 if proba > 0.5 else 0, out.detach().numpy())))).double()
    else:
        preds = torch.argmax(out, dim=1).numpy()
    return f1_score(yb.numpy(), preds, average='weighted')

def get_data_batches(X_train, y_train, X_val, y_val, bs):
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    return (
        DataLoader(train_data, batch_size=bs, shuffle=True),
        DataLoader(val_data, batch_size=bs, shuffle=True),
    )

def create_multiclass_labels(rel_sents, no_rel_sents, n_sents_task, task):
    labels = torch.zeros(n_sents_task, dtype=torch.long) if task == 'task2' else torch.ones(n_sents_task, dtype=torch.long) * 2
    for i in range(n_sents_task):
        if i in rel_sents:
            labels[i] += 1 
    return labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax