"""
Exploring CovNN using PyTorch.
Coded from scratch after reading several kaggle kernles.
-------------------
Future TODO:
1. Clean up the code
2. Bring in ensemble
3. Try use transfer learning
-------------------
Dataset:
MNIST: https://www.kaggle.com/c/digit-recognizer/data
-------------------
"""

############################################
# Imports
############################################
import pandas as pd  # database
import numpy as np  # math
import matplotlib.pyplot as plt  # plot
# from matplotlib import colors
from datetime import datetime
import time

from sklearn.decomposition import PCA  # PCA
from mpl_toolkits.mplot3d import Axes3D  # 3d plot
from sklearn.manifold import TSNE

import torchvision  # built-in datasets

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


############################################
# Data preprocessing
############################################


class DigitDataset(Dataset):
    """ Dataset class for digits.
    REF: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, dataframe, type, transform=None):
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
    assert (len(images) == len(labels) and (
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
# The Neural Net
############################################


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


def train(epoch, model, train_loader, exp_lr_scheduler, optimizer, criterion):
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


def evaluate(data_loader, model):
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


def prediciton(data_loader, model):
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

    if phase >= 2:  # show data augmentation
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

    if phase >= 3:  # Visualizing classes
        if verbose:
            print("============= PHASE 3: VISUALIZATION CLASSES")
        X_train = train_df.iloc[:, 1:].values
        y_train = train_df.iloc[:, 0].values
        plot_PCA(X_train, y_train, display=show_img)
        plot_tSNE(X_train, y_train, display=show_img, pca_dim=50, shrink=True)
        if verbose:
            print("============= END OF PHASE 3")
        if phase == 3:
            exit(0)

    if phase >= 4:  # Load the data (finally......)
        if verbose:
            print("============= PHASE 4: Load the data")

        batch_size = 64
        if verbose:
            print("Set batch size to 64.\n")

        # Train Set
        train_transform_f = transforms.Compose(
            [transforms.ToPILImage(), transforms.RandomRotation(degrees=20), transforms.RandomAffine(
                degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),
             transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

        train_dataset = DigitDataset(train_df, 'train', train_transform_f)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True)

        if verbose:
            print("Training set: dataset created, loader created.\n")

        # Test Set
        test_transform_f = transforms.Compose([transforms.ToPILImage(
        ), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

        test_dataset = DigitDataset(test_df, 'test', test_transform_f)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False)
        if verbose:
            print("Testing set: dataset created, loader created.\n")

        # Validation Set
        valid_transform_f = transforms.Compose([transforms.ToPILImage(
        ), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

        valid_dataset = DigitDataset(valid_df, 'valid', test_transform_f)

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset, batch_size=batch_size, shuffle=False)
        if verbose:
            print("Validation set: dataset created, loader created.\n")
            print("============= END OF PHASE 4")

        if phase == 4:
            exit(0)

    if phase >= 5:
        if verbose:
            print("============= PHASE 5: Init the neural network!")

        model = ConvNet()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        criterion = nn.CrossEntropyLoss()

        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1)

        if verbose:
            print("============= END OF PHASE 5")

        if phase == 5:
            exit(0)

    if phase >= 6:
        if verbose:
            print("============= PHASE 6: Train and Test")

        n_epochs = 1
        for epoch in range(n_epochs):
            train(epoch, model, train_loader,
                  exp_lr_scheduler, optimizer, criterion)

            evaluate(train_loader, model)

        test_pred = prediciton(test_loader, model)

        out_df = pd.DataFrame(np.c_[np.arange(1, len(
            test_dataset) + 1)[:, None], test_pred.numpy()], columns=['ImageId', 'Label'])

        out_df.head()

        out_df.to_csv('result.csv', index=False)

        if verbose:
            print("============= END OF PHASE 6\n============= END OF PROGRAM")


if __name__ == '__main__':
    print(
        "Phases:\n\t 1. expore the dataset\n\t 2. on data augmentation\n\t 3. visualizing datasets\n\t 4.load the data\n\t 5.Init the neural net \n\t 6. Train and Test!")
    phase = input("key in phase number 1,2,...,6: ")
    main(int(phase))
