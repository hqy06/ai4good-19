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
        # assert (
        #     (type == 'valid' or 'test') and transform is not None), "only trainning data can be augmented!"
        self.transform = transform

        self.n_features = 784  # 28*28-dim input
        self.n_supports = 1  # 1-dim output

        if type == 'test':
            self.X = dataframe.values.reshape(
                (-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = None
        else:
            self.X = dataframe.iloc[:, 1:].values.reshape(
                (-1, 28, 28)).astype(np.uint8)
            self.y = dataframe.iloc[:, 0].values

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if self.y is not None:
            return self.transform(self.X[index]), self.y[index]
        else:
            return self.transform(self.X[index])


# %%
batch_size = 64
train_transform_f = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation(degrees=20), transforms.RandomAffine(
    degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

train_dataset = DigitDataset(train_df, 'train', train_transform_f)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Test Set
test_transform_f = transforms.Compose([transforms.ToPILImage(
), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

test_dataset = DigitDataset(test_df, 'test', test_transform_f)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)


# %%
# Define a convolution neural net
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.Dropout(0.25),
                                  nn.Conv2d(64, 32, 3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, padding=1),
                                  nn.Dropout(0.20)
                                  )

        self.fc = nn.Sequential(nn.Linear(32 * 8 * 8, 500),
                                nn.ReLU(),
                                nn.BatchNorm1d(500),
                                nn.Dropout(p=0.5),
                                nn.Linear(500, 100),
                                nn.ReLU(),
                                nn.Linear(100, 10),
                                nn.ReLU(),
                                nn.Softmax(dim=1)
                                )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        return self.fc(x)


# %%
# Init the neural net
model = ConvNet()
optimizer = optim.Adam(model.parameters(), lr=0.003)

criterion = nn.CrossEntropyLoss()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# %%
# Define Train and Evaluation function


def train(epoch):
    model.train()
    exp_lr_scheduler.step()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * len(
                data), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.data.item()))


def evaluate(data_loader):
    model.eval()
    loss = 0
    correct = 0

    for data, target in data_loader:
        # data, target=Variable(data, volatile=True), Variable(target)
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        # if torch.cuda.is_available():
        #     data=data.cuda()
        #     target=target.cuda()

        output = model(data)

        loss += F.cross_entropy(output, target, size_average=False).data.item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(data_loader.dataset)

    print('\nAverage loss: {:.4f}, Accuracy: {}/{}({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


n_epochs = 1
for epoch in range(n_epochs):
    train(epoch)
    evaluate(train_loader)


def prediciton(data_loader):
    model.eval()
    test_pred = torch.LongTensor()

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data = Variable(data)

        # data=Variable(data, volatile=True)

        output = model(data)

        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)

    return test_pred


test_pred = prediciton(test_loader)


out_df = pd.DataFrame(np.c_[np.arange(1, len(
    test_dataset) + 1)[:, None], test_pred.numpy()], columns=['ImageId', 'Label'])

out_df.head()

out_df.to_csv('submission.csv', index=False)
