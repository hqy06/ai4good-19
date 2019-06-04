# ai4good-19
This is the repo for my own work inspired by the intensive training lectures and tutorials during the AI4Good summer lab at MILA. However, other than the warm-up exercise, NO LAB MATERIAL WILL BE SHARED HERE.

Here is summay of toy-porjects in this repo:
- kNN with IRIS using sklearn from scratch
- KMean with WINE using sklearn from scratch
- MLP classifier practices following the official sklearn tutorial [1](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py). [2](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html#sphx-glr-auto-examples-neural-networks-plot-mlp-alpha-py), [3](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py)
- MLP regresser practice (no reference other than matplotlib stuffs)
- CNN using PyTorch from scratch, use [this kaggle tutorial](https://www.kaggle.com/puneetgrover/training-your-own-cnn-using-pytorch/comments) as reference
- vanilla RNN name classifier from scratch using PyTorch, using [the official pytorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) as read-ahead reference
- writer RNN with LSTM cells using PyTorch, using [this keras tutorial](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/) as read-ahead reference.


Currently working on the writer RNN.
I omit all GPU stuffs because I want to test and run all these programs on my own laptop quickly.
All resources, other than the official documentation of python libraries, ever used or refered to are contained in the markdown files.

### About the datasets folder:

The `names` are from the pytorch tutorial.

The `books` are downloaded from the [Project Gutenberg](https://www.gutenberg.org/ebooks/). The non-text parts of the original text files are manually deleted by me as I don't want to spend too long time doing io manipulation. There is also a foo text for function testing.

The `digits` dataset is from kaggle, named [digit recognizer](https://www.kaggle.com/c/digit-recognizer).
