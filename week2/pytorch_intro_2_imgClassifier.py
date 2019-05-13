"""
Toying an image classifier with CIFAR10 following the official tutorial of PyTorch
--------------------------------
About CIFAR10:
10 classes
(RGB) 3-channel color image
size 32*32 pixels
--------------------------------
Things to do:
 - load and normalize data using `torchvision`
 - define a CNN
 - define a loss function
 - train the network on the training data
 - test the network on the test data
"""


import torch
import torchvision
import torchvision.transforms as transforms
