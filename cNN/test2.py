# %% Import Block
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

import math
import random

from PIL import Image, ImageOps, ImageEnhance
import numbers

import matplotlib.pyplot as plt

# %% Import csv
train_df = pd.read_csv('../datasets/digits_train.csv')
test_df = pd.read_csv('../datasets/digits_test.csv')
# Sanity check
test_df.shape
train_df.shape

# %% Dataset Class


class DigitDataset(Dataset):
    """ Dataset class for digits.
    REF: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, dataframe, type,  transform=None):
        """self.X are self.y are of type numpy.ndarray"""
        assert (
            type == 'train' or 'test' or 'valid'), "type must be train, valid, or test"
        self.type = type
        self.n_samples = dataframe.shape[0]
        assert (
            (type == 'valid' or 'test') and transform is not None), "only trainning data can be augmented!"
        self.transform = transform

        self.n_features = 784  # 28*28-dim input
        self.n_supports = 1  # 1-dim output

        self.X = dataframe.iloc[:, 1:].values.reshape(
            (-1, 28, 28)).astype(np.uint8)
        self.y = dataframe.iloc[:, 0].values

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if self.type is 'train':
            return self.transform(self.X[index]), self.y[index]
        else:
            return self.X[index], self.y[index]


# %%
batch_size = 64
train_transform_f = transforms.Compose(
    [transforms.ToPILImage(), transforms.RandomRotation(degrees=20), transforms.RandomAffine(
        degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),
     transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

train_dataset = DigitDataset(train_df, 'train', train_transform_f)


train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)


test_dataset = DigitDataset(test_df, 'test', transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,)]))
