"""
Exploring CovNN using PyTorch.
Coded from scratch after reading several kaggle kernles.
-------------------
Tasks:
0. import and/or load data
1. preprocessing data
2. visualizing dataset in 3D
3. define a cNN and train it
4. bring in ensembel!
5. visualization: img through cNN layers.
6. download the prediction file
-------------------
Dataset:
MNIST: https://www.kaggle.com/c/digit-recognizer/data
"""


############################################
# Imports
############################################
import pandas as pd     # database
import numpy as np      # math
import matplotlib.pyplot as plt    # plot

import torch            # torch
import torchvision      # built-in datasets
import torch.nn as nn   # neural net
import torch.optim as optim     # optimizer
import torch.nn.functional as F  # functionsals
from torch.autograd import Variable  # for backpropagation
from torchvision import transforms

############################################
# Data preprocessing
############################################


############################################
# Ploting utilites
############################################
from datetime import datetime


def show_images(images, fig_size, labels, predictions=None, cols=None, name=None):
    """
    Display multiple (>=2) images of given size.
    ----
    :param images: a list of images of type=numpy array
    :param fig_size: numpy shape of each image
    :param labels: a list of labels
    :param predictions: default to None, prediction made by the model
    :param cols: specifying the number of columns, default is none for automatic arrange
    :param name: name of the saved png file, if not specified, used date time
    """
    assert(len(images) == len(labels) and (
        (predictions is None) or len(labels) == len(predictions)))

    n_images = len(images)
    # Set titles
    if predictions:
        titles = ['label={}, pred={}'.format(
            l, p) for l, p in zip(labels, predictions)]
    else:
        titles = ['label={}'.format(l) for l in labels]

    # Set canvas arrangement
    if cols:
        n_col = int(cols)
        n_row = int(np.ceil(n_images / float(n_col)))
    else:
        n_col = int(np.floor((n_images * 2) ** 0.5))
        n_row = int(np.ceil(n_images / float(n_col)))

    print('# col={}, # row={}'.format(n_col, n_row))

    # Set output file name
    if name:
        f_name = '%s.png' % (name)
    else:
        f_name = '%s.png' % (datetime.now().strftime('%Y%m%d-%H%M'))

    fig, axes = plt.subplots(n_row, n_col, figsize=fig_size)

    axis = axes.ravel()

    for img, ax, title in zip(images, axis, titles):
        print('img of shape {}'.format(img.shape))
        ax.imshow(img)
        ax.set_title(title)

    fig.tight_layout()
    plt.savefig(f_name)
    plt.show(block=False)
    plt.pause(3)
    plt.close()


############################################
# Main
############################################


def main(verbose=True, show_img=True):
    # Data preprocessing
    train_df = pd.read_csv('../datasets/digit_train.csv')
    test_df = pd.read_csv('../datasets/digit_test.csv')
    n_train = train_df.shape[0]
    n_test = test_df.shape[0]
    assert (test_df.shape[1] == train_df.shape[1]
            ), 'Number of features mismatch!'
    n_features = train_df.shape[1] - 1

    if verbose:
        print('****** Exploring dataset\n')
        print('DataFrame: TRAIN, %d samples, labeled, each with %d features' %
              (n_train, n_features))
        print('DataFrame: TEST, %d samples, labeled, each with %d features' %
              (n_test, n_features))
        print('\nA small sample from TRAIN:')
        print(train_df.head())

    if show_img:
        rand_train = np.random.randint(n_train, size=9)
        images = train_df.iloc[rand_train, 1:].values
        labels = train_df.iloc[rand_train, 0]
        show_images(images, labels, pred=None)


if __name__ == '__main__':
    main()
