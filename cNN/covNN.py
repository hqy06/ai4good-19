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
import time

from sklearn.decomposition import PCA                 # PCA
from mpl_toolkits.mplot3d import Axes3D  # 3d plot
from sklearn.manifold import TSNE

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


def show_images(images, fig_size, labels, predictions=None, cols=None, name=None, display=True):
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
    if display:
        plt.show(block=False)
        plt.pause(3)
    plt.close()


def plot_PCA(data, labels, display=True, dim=3):
    assert (dim == 2 or dim == 3)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data)

    pca_1 = pca_result[:, 0]
    pca_2 = pca_result[:, 1]
    pca_3 = pca_result[:, 2]

    if dim == 3:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.gca(projection='3d')
        dots = ax.scatter(pca_1, pca_2, pca_3, c=labels,
                          cmap='tab10', alpha=0.2)
        plt.colorbar(dots)
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')
        ax.set_title('visualization of training set using PCA in 3D', size=12)
    elif dim == 2:
        fig = plt.figure(figsize=(16, 10))
        ax = plt.gca()
        dots = ax.scatter(pca_1, pca_2,
                          c=labels, cmap="tab10", alpha=0.3)
        plt.colorbar(dots)
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_title('visualization of training set using PCA in 2D', size=12)
    plt.savefig('visualization_train_PCA.png')
    # fig.tight_layout()
    if display:
        plt.show(block=False)
        plt.pause(2)
    plt.close()


def plot_tSNE(data, labels, display=True, pca_dim=None, shrink=True, verbose=True):
    # sanity check
    assert (data.shape[0] == labels.shape[0])

    # No more than 10000 samples allowed
    n_samples = data.shape[0]
    if n_samples > 10000 and shrink:
        if verbose:
            print("More than 10k samples! ramdom sample from the original dataset")
        rand_indices = np.random.randint(n_samples, size=10000)
        data = data[rand_indices, :]
        labels = labels[rand_indices]

    if pca_dim:
        if verbose:
            print("Let's do PCA to find the {} principal components".format(pca_dim))
        pca = PCA(n_components=pca_dim)
        data = pca.fit_transform(data)
        if verbose:
            print('Cumulative explained variation for {} principal components: {}'.format(
                pca_dim, np.sum(pca.explained_variance_ratio_)))

    # Let's do t_SNE!
    v = 0
    if verbose:
        v = 1
    tsne = TSNE(n_components=2, verbose=v, perplexity=40, n_iter=300)
    time_start = time.time()
    tsne_results = tsne.fit_transform(data)
    if verbose:
        print('\nt-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    # Plot the result!
    fig = plt.figure(figsize=(16, 10))
    ax = plt.gca()
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels,
               cmap="tab10", alpha=0.3)
    ax.set_title('visualization of training set using t-SNE', size=12)
    plt.savefig('visualization_train_tSNE.png')

    if display:
        plt.show(block=False)
        plt.pause(2)
    plt.close()

############################################
# Main
############################################


def main(phase, verbose=True, show_img=True):
    # ====== Load and Explore Data =======
    train_df = pd.read_csv('../datasets/digits_train.csv')
    test_df = pd.read_csv('../datasets/digits_test.csv')
    assert (test_df.shape[1] == train_df.shape[1] - 1
            ), 'Number of features mismatch! train={} <> 1+test={}'.format(train_df.shape[1], test_df.shape[1])
    n_features = train_df.shape[1] - 1
    if verbose:
        print("****** dataset read from csv files")

    if phase >= 1:  # Explore the dataset
        if verbose:
            print("============= PHASE 1: explore dataset")
            print('DataFrame: TRAIN, %d samples, labeled, each with %d features' %
                  (train_df.shape[0], n_features))
            print('DataFrame: TEST, %d samples, labeled, each with %d features' %
                  (test_df.shape[0], n_features))
            print('\nA small sample from TRAIN:')
            print(train_df.head())

        # Display random digits
        if show_img:
            rand_train = np.random.randint(test_df.shape[0], size=9)
            images = train_df.iloc[rand_train, 1:].values.reshape(-1, 28, 28)
            labels = train_df.iloc[rand_train, 0]
            show_images(images, (28, 28), labels, cols=3)

        # Display a histogram
        plt.hist(train_df.iloc[:, 0], color="skyblue",
                 ec="cornflowerblue", lw=2)
        plt.title('histogram on train dataset')
        plt.savefig('histogram.png')
        if show_img:
            plt.show(block=False)
            plt.pause(2)
        plt.close()

        if verbose:
            print("============= END OF PHASE 1")

        if phase == 1:
            exit(0)

    # ====== Preprocess the Data =======
    # Seperate the validation set
    train_df, valid_df = split_dataframe(train_df, fraction=0.9)
    if verbose:
        print("****** Seperated validation set from training set.")

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

    if phase >= 2:   # show data augmentation
        if verbose:
            print("============= PHASE 2: illustrate data augmentation")

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

        show_images(images, (28, 28), t_names, display=show_img, cols=3)
        if verbose:
            print("============= END OF PHASE 2")

        if phase == 2:
            exit(0)

    if phase >= 3:   # Visualizing classes
        if verbose:
            print("============= PHASE 3: VISUALIZATION CLASSES")
        X_train = train_df.iloc[:, 1:].values
        y_train = train_df.iloc[:, 0].values
        plot_PCA(X_train, y_train, display=show_img)
        plot_tSNE(X_train, y_train, display=show_img, pca_dim=50, shrink=True)
        if verbose:
            print("============= END OF PHASE 3")


if __name__ == '__main__':
    print("Phases:\n\t 1. expore the dataset\n\t 2. on data augmentation")
    phase = input("key in phase number 1, 2, or 3: ")
    main(int(phase))
