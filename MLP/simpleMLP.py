

"""
Just a very simple fully-connected feed-forward neural network SKELETON.

No testing, no debuggin, it won't run and is not supposed to work a functional unit as sklearn and pytorch have already done a brilliant job.

For futher reading, check these two websites:
- https://github.com/meetvora/mlp-classifier/blob/master/neuralnet.py
- http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

    O ...... O              input layer      {X}    ...

  o ........... o           1st hidden layer {H1}   ...

      o....o                2nd hidden layer {H2}   ...

  O ............O           output layer     {Y}    ...
"""

import scipy
import numpy as np


################################################################################

class simplyMLP:
    def __init__(self, input_size, output_size, output_act, h1_size, h1_act, h2_size, h2_act):
        """ store all the hyperparamters and randomly init weight and bias"""
        # A python list storing number of neurons in each layer
        self.input_size = input_size
        self.sizes = [input_size, h1_size, h2_size, output_size]
        self.depth = self.sizes.shape[0] - 1

        # The activation functions. Be aware that the input doesn't need activation function
        self.acts = [select_act(h1_act), select_act(
            h2_act), select_act(output_act)]
        self.loss = [select_loss(h1_act), select_loss(
            h2_act), select_loss(outpu_act)]

        # Other hyperparamters, including learning rate and choice of regularization
        # self.eta = learning_rate
        # self.reg_lambda = reg_lambda
        # self.reg_method = reg_method      # must be from 0,1,2 (no-op, L1, L2)
        # Moved to the batch_update function

        # Randomly init the parameters
        self.bias, self.weight = self._init_parameters()

        # Init the cost list
        self.costs = self._init_costs()

    def _init_parameters(self):
        """
        Randomly init theta, aka the weight matrix plus the bias vector. Since the input doesn't count as a layer, no [1:] slice or -1 in depth
        """
        # A list of weight vecotrs, each weight vector is of type np.ndarray
        # Notice the python-ish in the line below
        weight = [np.random.randn(layer_size, layer_input_size)
                  for layer_size, layer_input_size in zip(self.sizes[1:], self.sizes[:-1])]

        # Bias also stored in a python list
        bias = [np.random.randn(layer_size, 1)
                for layer_size in self.sizes[1:]]

        return weight, bias

    # def calculate_loss(self, xx):
    #     for layer in range(self.depth):
    #         calculate_layer_loss()

    def fit_mini_batch(self, training_data, mini_batch_size=32, epochs=500, eta=0.001, reg_rate=0, reg_method=0):
        """Fit the parameters (w,b) with the training data, do minibatches as indicated"""
        for epoch in range(epochs):
            mini_batches = self._generate_mini_batches(
                training_data, mini_batch_size)
            for mini_batch in mini_batches:
                self.batch_update(mini_batch, n=training_data.shape[0])

    def _generate_mini_batches(self, training_data, mini_batch_size):
        epoch_data = shuffle_data(training_data)
        mini_batches = [epoch_data[k:k + minibatches]
                        for k in range(0, training_data.shape[0], mini_batch_size)]
        return mini_batches

    def batch_update(self, mini_batch, n):
        """ One time mini_batch update with GD"""
        bb = [np.zeros(b.shape) for b in self.bias]
        ww = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            d_bb, d_ww = self.backprop(x, y)
            bb = [b + db for b, db in zip(bb, d_bb)]
            ww = [w + dw for w, dw in zip(ww, d_ww)]

        # update bias
        # b_new = b_old - eta * (b_batch / batch_size)
        self.bias = [bias - eta * b / len(mini_batch)
                     for bias, b in zip(self.bais, bb)]

        if reg_method == 1:
            # TODO: update weight with L1 penalty
            # CF: https://stackoverflow.com/questions/44621181/performing-l1-regularization-on-a-mini-batch-update

        elif reg_method == 2:
            self.weights = [weight - eta * w / len(
                mini_batch) - eta * reg_rate * weight / n for weight, w in zip(self.weights, ww)]
        else:
            raise ValueError('reg_method should be either 1 or 2')

    def backprop(self, x, y):
        """Gradient for cost function and backprop to the input layer"""

        nabla_b = [np.zeros(b) for b in self.bias]
        nabla_w = [np.zeros(w) for w in self.weights]

        layers_zs = []
        layer_io = x
        for b, w, act in zip(self.bias, self.weights, self.acts):
            z = b + np.dot(w, layer_io)
            layers_zs.append(z)
            layer_io = act(z)

        delta = layer_io[-1] - y
        nablas_b[-1] = delta
        nabla_w[-1] = np.dot(detlta, layer_io[-2].transpose())
        for level in range(2, self.depth - 1, 0, -1):
            delta = np.dot(self.weights[level + 1].transpose(), delta) *
            cost_derivative(layers_zs[level], self.acts[level])
            nabla_b[-level] = delta
            nabla_2[-level] = np.dot(delta, self.acts[-level - 1].transpose)
        return nabla_b, nabla_w


################################################################################
# GD related
################################################################################

def cost_derivative(layer_io, activation_fn):
    if activation_fn == sigmod:
        return sigmod(layer_io) * (1 - sigmod(layer_io))
    elif activation_fn == softmax:
        return softmax(layer_io) * (1 - softmax(layer_io))
    else:
        raise ValueError(
            'No such activation function, use sigmoid or softmax!')


################################################################################
# Activation function & loss function for neurons
################################################################################


def select_act(act_type):
    # TODO: select activation function according to keyword
    if act_type == 'sigmod':
        return sigmoid
    elif act_type == 'softmax':
        return softmax
    else:
        raise ValueError(
            'unexpected activation function, use sigmoid or softmax')


def select_loss(act_type):
    if act_type == 'sigmod':
        return cross_entropy
    elif act_type == 'softmax':
        return log_likelihood
    else:
        raise ValueError(
            'unexpected activation function, use sigmoid or softmax')


def softmax(x):
    """
    A generalized version of logistic activation function
    """
    scipy.special.softmax(x)


def sigmod(x):
    """
    The logistic activation function that is also called "sigmod"
    """
    return 1 / (1 + np.exp(-x))
#
#
# def relu(x):
#     """
#     The classic activation function that is widely used
#     """
#     return np.max(0, x)
#
#
# def identity(x):
#     """
#     The no-op activation function, usually for the bottle-neck layers
#     """
#     return x
#
#
# def tanh(x):
#     """
#     Just the hyperbolic tan
#     """
#     return np.tanh(x)

# Error functions


# def mse(x):
#     """ associate with identity/linear"""
#     pass


def cross_entropy(x):
    """ associate with sigmoid, aka logistic"""
    pass


def log_likelihood(x):
    """ associate with softmax"""
    pass

###############################################################################
# Manipulate the data set
###############################################################################


def load_data():
    pass


def train_test_split():
    pass


def shuffle_data():
    pass
