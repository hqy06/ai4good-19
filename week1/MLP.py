# A neural network from scratch

import numpy as np
import scipy
from sklearn import datasets


class hidden_layer:
    pass


class outpu_layer:
    pass


class mlp:

    def __init__(self, struct, reg=0):
        """
        Initialize a fully-connected, feed-forward neural network.
        @para: struct
            [[10,'s'],[3,'r'],[5,'l']]
        index=0: the number of neurons for each layer
        index=1: the activation function for each layers, this also determines the loss function for this layers.
        @para: reg
        For the regularization, 0 for none, 1 for L1, 2 for L2
        """
        # Anything else
        self.depth = len(struct)
        self.struct = struct
        # Randomly init the weight matrix W and bias vector b
        self.weights, self.bias = self.rand_init_para(struct)

    def init_network(self, struct):
        while len(struct) != 1:
            h = struct.pop(0)
            self.init_hidden_layer(entry)
        last = struct.pop(0)
        self.init_output_layer(entry)
        return weights, bias

    def train():
        pass


def load_data():
    # https://scikit-learn.org/stable/datasets/index.html#wine-recognition-dataset
    # https://machinelearningmastery.com/generate-test-datasets-python-scikit-learn/
    pass


################################################################################
# The Activation Functions
################################################################################

def sigmod(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.max(0, z)


def softmax(z):
    return scipy.special.softmax(z)
