# -*- coding: utf-8 -*-

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

try:
    import skdim
except ImportError:
    os.system("pip install scikit-dimension")
    import skdim


def invert_binary_label(x):
    return 0. if x == 1. else 1.


def add_label_noise(labels, label_noise=None):
    if isinstance(label_noise, float) and 0 < label_noise <= 1:
        n_noisy = int(len(labels) * label_noise)
        print(f"Inverting {n_noisy} out of {len(labels)} sample labels")
        indices = torch.range(0, len(labels) - 1)
        to_invert = torch.multinomial(indices, n_noisy, replacement=False)
        labels[to_invert] = torch.tensor(
            [invert_binary_label(t) for t in labels[to_invert]]
        )
    return labels


def make_data(
    n_samples, n_dim, n_intrinsic_dim, coefs=None, label_noise=None
):
    """Make multivariate Gaussian data with two classes.

    The class of a sample is 0 if it lies on the negative side of a linear
    hyperplane and 1 otherwise. If `label_noise` is not None, it must be
    in the range [0, 1], and that fraction of the labels are randomly inverted.
    """
    manifolds = skdim.datasets.BenchmarkManifolds(random_state=0)
    data = manifolds.generate(
        name="M12_Norm",
        n=n_samples,
        dim=n_dim,
        d=n_intrinsic_dim
    )
    input = torch.tensor(data).float()

    if coefs is None:
        coefs = torch.randn(n_dim)
    target = (input * coefs).sum(dim=1) > 0
    target = target.float()
    target = add_label_noise(target, label_noise=label_noise)
    target = target.reshape((n_samples, 1))

    return input, target


def make_data_nonlinear(
    n_samples:int, n_dim:int, i_dim:int, label_noise=None
    ):
    """Make multivariate Gaussian data with two classes.

    The class of a sample is determined by whether it's on the positive or
    negative side of a linear hyperplane.
    """
    positive_outcome_ratio = 0.5
    benchmark = skdim.datasets.BenchmarkManifolds(random_state=0)

    n_sample = int(n_samples * (1 - positive_outcome_ratio))
    p_sample = int(n_samples * positive_outcome_ratio)

    data1 = benchmark.generate(name="M12_Norm", n=n_sample, dim=n_dim, d=i_dim)
    data2 = benchmark.generate(name="M7_Roll", n=p_sample, dim=n_dim, d=i_dim)

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'Scatter3d'}] * 2])

    trace1 = go.Scatter3d(dict(zip(['x','y','z'], data1.T[:3])),
            mode='markers', marker=dict(size=1.5,colorbar=dict()))
    trace2 = go.Scatter3d(dict(zip(['x','y','z'], data2.T[:3])),
            mode='markers', marker=dict(size=1.5, colorbar=dict()))

    fig.add_traces([trace1, trace2], rows=1, cols=[1,2])
    fig.layout.update(height=450, width=800)

    input = torch.tensor(np.vstack((data1, data2))).float()
    target = torch.tensor([0.] * len(data1) + [1.] * len(data2))
    target = add_label_noise(target, label_noise=label_noise)
    # The target must be 2-dimensional.
    target.unsqueeze_(-1)

    return input, target, fig


def get_twonn_dim(data):
    """Get the intrinsic dimensionality of the data using Two NN.
    """
    two_nn = skdim.id.TwoNN().fit(data)
    return two_nn.dimension_


def train_val_test_split(input, target, batch_size):
    """split data into train, val, test with ratio 8:1:1 and return as DataLoader class
      """
    input = input.clone()
    target = target.clone()

    dataset = TensorDataset(input, target)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    return(train_loader,val_loader,test_loader)


class MLP(nn.Module):
    """
    Make a one hidden-layer MLP with an optional ReLU non-linearity.
    """
    def __init__(self, n_input, n_hidden, use_relu=True):
        super(MLP, self).__init__()
        self.use_relu = use_relu
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.fc1(x)
        if self.use_relu:
          x = self.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class LogisticRegression(nn.Module):
    """Make a simple LR as MIA model
      """
    def __init__(self,n_input):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(in_features=n_input, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


def generate_mia_data(train_loader,val_loader,mlp):
    """
    Generate data for MIA model with train and val DataLoader and trained MLP.
    """
    i=0
    for batch_data in train_loader:
      batch_inputs = batch_data[0]
      batch_outputs = mlp(batch_inputs)
      if i == 0:
        train_feature = batch_outputs.detach().numpy()
        train_label = np.ones(batch_data[0].shape[0])
      else:
        train_feature = np.vstack((train_feature, batch_outputs.detach().numpy()))
        train_label = np.concatenate((train_label, np.ones(batch_data[0].shape[0])))
      i+=1

    i=0
    for batch_data in val_loader:
      batch_inputs = batch_data[0]
      batch_outputs = mlp(batch_inputs)
      if i == 0:
        val_feature = batch_outputs.detach().numpy()
        val_label = np.zeros(batch_data[0].shape[0])
      else:
        val_feature = np.vstack((val_feature, batch_outputs.detach().numpy()))
        val_label = np.concatenate((val_label, np.zeros(batch_data[0].shape[0])))
      i+=1

    random_indices = np.random.choice(train_feature.shape[0],
                                    size=val_feature.shape[0],
                                    replace=False)
    features = np.vstack((train_feature[random_indices, :], val_feature))
    labels = np.concatenate((train_label[random_indices], val_label))
    return(features, labels)


# Functions for train the MIA LR model and compute CV AUC, will convert into a class latter
def train(model, train_loader, criterion, optimizer):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            y_true.extend(labels.numpy())
            y_pred.extend(outputs.numpy().flatten())
    auc = roc_auc_score(y_true, y_pred)
    return auc


def cross_validation(model, data, target, n_splits, lr=0.01, epochs=30):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in cv.split(data):
        train_data = torch.tensor(data[train_idx]).float()
        train_target = torch.tensor(target[train_idx]).float()
        test_data = torch.tensor(data[test_idx]).float()
        test_target = torch.tensor(target[test_idx]).float()

        train_dataset = TensorDataset(train_data, train_target)
        test_dataset = TensorDataset(test_data, test_target)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model.train()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            train(model, train_loader, criterion, optimizer)

        model.eval()
        auc = test(model, test_loader)
        aucs.append(auc)

    return sum(aucs) / n_splits

