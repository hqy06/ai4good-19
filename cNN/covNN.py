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
from matplotlib import colors
from datetime import datetime

import torch            # torch
import torchvision      # built-in datasets
import torch.nn as nn   # neural net
import torch.optim as optim     # optimizer
import torch.nn.functional as F  # functionsals
from torch.utils.data import Dataset, DataLoader    # ud dataset
from torch.autograd import Variable  # for backpropagation
from torchvision import transforms

############################################
# Data preprocessing
############################################


def DigitDataset(Dataset):
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


def split_dataframe(dataframe=None, fraction=0.9, rand_seed=42):
    """Shuffle and split the dataframe"""
    df_1 = dataframe.sample(frac=fraction, random_state=rand_seed)
    df_2 = dataframe.drop(df_1.index)
    return df_1, df_2

############################################
# Ploting utilites
############################################


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
    # ====== Load and Explore Data =======
    train_df = pd.read_csv('../datasets/digits_train.csv')
    test_df = pd.read_csv('../datasets/digits_test.csv')
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
        # Display random digits
        rand_train = np.random.randint(n_train, size=9)
        images = train_df.iloc[rand_train, 1:].values
        labels = train_df.iloc[rand_train, 0]
        show_images(images, (28, 28), labels, pred=None)

        # Display a histogram
        plt.hist(train_df.iloc[:, 0], color="skyblue",
                 ec="cornflowerblue", lw=2)
        plt.title('histogram on train dataset')
        plt.save('histogram.png')
        plt.show()
        plt.pause(2)
        plt.close()

    # ====== Preprocess the Data =======
    # Seperate the validation set
    train_df, valid_df = split_dataframe(train_df, fraction=0.9)

    """ The following code snipet didn't make use of PyTorch, need to define a custom dataset class
    # Extract input matrix and target vector for each set
    X_train, y_train = train_df.values[:, 1:], train_df.values[:, 0]
    X_valid, y_valid = valid_df.values[:, 1:], valid_df.values[:, 0]
    X_test, y_test = test_df.values[:, 1:], test_df.values[:, 0]

    # Normalize the data (both the training set and the validation set)
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_valid = (X_valid - X_train.mean()) / X_valid.std()"""

    # Prepare the datasets
    batch_size = 64
    RandAffine = transforms.RandomAffine(
        degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2))
    train_transform_f = transforms.Compose([transforms.ToPILImage(
    ), RandAffine, transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

    if show_img:    # show data augmentation
        rotate = transforms.RandomRotation(degrees=45)
        shift = RandAffine
        composed = transforms.Compose([rotate, shift])

        rand_indices = np.random.randint(train_df.shape[0], size=3)
        samples = train_df.iloc[rand_indices,
                                1:].values.reshape(-1, 28, 28).astype(np.uint8)

        t_samples, t_names = [], []
        for s in samples:
            img = transforms.ToPILImage()(s)
            for t in [rotate, shift, composed]:
                t_samples.append(np.array(t(img)))
                t_names.append(type(t).__name__)

        show_images(images, (28, 28), titles)

    # ===========


if __name__ == '__main__':
    main()
