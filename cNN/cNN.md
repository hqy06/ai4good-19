# CovNN: PyTorch Basic

**Check Onenote notebook _19w_ for study notes for theoretical perspective on convolutional neural nets.**

When we want to play with neural nets, it is better to switch from `sk-learn` to `PyTorch` as the latter provide GPU solution, which is essential when training large scale neural net. (And you don't want to write the backpropagation yourself again)

## The Datasets

### Toy image datasets from `torchvision`

| Dataset | Type           | Comment                                                    |
| ------- | -------------- | ---------------------------------------------------------- |
| MNIST   | labeled images | A bench mark for ML, simple and out-of-date for serious ML |
| CIFAR10 | labeled images | A 10-class version of "80 million tiny images "            |
| STL-10  | labeled images | adapted from CIFAR10 for unsueprvised learning             |

We have a [tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.htm) for this kind!

### Make use of `sk-learn`

Get `numpy` arrays following the normal `sklearn` procedure, then convert them to `PyTorch` tensors.

### Kaggle CVS

Normal CVS approach, come back later!

Ref: documentation [here](https://pytorch.org/docs/stable/torchvision/datasets.html)

## Exploring CNN in PyTorch

### Resources

CNN tutorials on Kaggle:

- by Puneet Grover: https://www.kaggle.com/puneetgrover/training-your-own-cnn-using-pytorch/comments
- by Bonhart: https://www.kaggle.com/bonhart/pytorch-cnn-from-scratch/notebook
- by Ryan Chang: https://www.kaggle.com/juiyangchang/cnn-with-pytorch-0-995-accuracy
- by Tony: https://www.kaggle.com/tonysun94/pytorch-1-0-1-on-mnist-acc-99-8/data

A nice PyTorch tutorial by yunjey on GitHub: https://github.com/yunjey/pytorch-tutorial

### Checklist

[ ] do you have `__main__`?
[ ] data loading and preprocessing
[ ] exploring the data: visualization, histogram, parameters
[ ] define a CovNN
[ ] Train and evaluation
[ ] Prediction on test set

### Nice tricks for python

Retrive working directory: `os.getcwd()` or `from pathlib import Path` and then `cwd = Path.cwd()`
