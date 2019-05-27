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

[x] do you have `__main__`?

[x] loading csv

[x] exploring the data: basic visualization, histogram, parameters

[x] Debugging and testing like a computer scientist (not a data scientist)

[x] Visualize multi-dim data: t-SNE

[x] the `Dataset` class in PyTorch

---

[x] ~~make a DataLoader~~ Nope, use PyTorch's built-in stuffs, be aware of function/class signatures!

[x] Load the data to the user-defined dataset class

[x] Define a neural net _PyTorch class_: from scratch OR as a child class of some existed CNN.

[ ] ~~Add ensemble! (AdaBoost or Bagging?)~~

[x] Training & Testing

[x] Save the result

### Nice tricks for python

Retrive working directory: `os.getcwd()` or `from pathlib import Path` and then `cwd = Path.cwd()`

To reload module:

```python3
# python3.x would require
# from importlib import reload
import X
reload( X )
from X import Y
```

Display multiple digist images:

- [stackoverflow Q&A](https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645)
- [by soply on github](https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1)

Why data augmentation matters:

- [medium blog](https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced)
- [standford paper](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
- [data augmentation an PyTorch](https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch)

Transfer Learning: you want a better init for you NN
https://cs231n.github.io/transfer-learning/

The transformation function in `torchvision.transforms` only works on PIL Image, rremember to convert them:

- PIL to numpy array: `np_img = np.array(pil_img)`
- numpy array to PIL: `pil_img = PIL.Image.fromarray(pn_img)` or `pil_img = transforms.ToPILImage()(np_img)'`

visualization: https://colah.github.io/posts/2014-10-Visualizing-MNIST/

Color reference: https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html

## Future work

Explore the "augmentation network" mentioned in standford's paper
Explore the "Transfer learning" (WOW also stanford)
Explore [this python package](https://github.com/mdbloice/Augmentor)
