"""
Toying a neural network according to the PyTorch tutorial
------------------------------------
CNN, 1 input image channel, 6 output channel, 5*5 convolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # convolution kernels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # affine operation: y=Wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        # all dimensions except the batch dim
        size = x.size()[1:]

        num_features = 1

        for s in size:
            num_features *= s

        return num_features

# Below is the console command for further exploration
# # Instantiation
# net = Net()
# print(net)
#
#
# # Dealing with parameters
# params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight
#
# # The forward pass
# input = torch.randn(1, 1, 32, 32)
# out = net(input)
# print(out)
#
# # Resizing, use the .view() function
# dummy = torch.randn(10)
# dummy.shape
# dummy
# re_dummy = dummy.view(1,-1)
# re_dummy.shape
#
# # Using the build-in loss functions
# criterion = nn.MSELoss()
# loss = criterion(net(input), re_dummy)
# print(loss)
