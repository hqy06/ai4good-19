# RNN: wow

## Good to Read

- PyTorch tutorials: [chatbot](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html), [char-level generation](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html), [char-level classification](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html), [Deep NLP](https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html), and [seq-to-seq](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- Christopher Olah's [blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) on LSTM
- Supplementary resource for Olah's tutorial: [1](https://skymind.ai/wiki/lstm)
- Attention mechanism in rNN: ()
- RNN with PyTorch: [jupyter notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb)
- Paper on document classification using rNN: [link](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
- K-Fold + Attention + LSTM: [kaggle kernel](https://www.kaggle.com/suicaokhoailang/10-fold-lstm-with-attention-0-991-lb)
- A fancy work to check back later: [link](https://github.com/MisterSchmitz/LSTM-Musician)

## Coding Practice

### Character-level name classification

Easy and vanilla. Aim at getting familiar with PyTorch's RNN ecology.

Steps to follow:

1. Read [this](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) PyTorch tutorial and understand the structure

2. Close the webpage and code from scratch, ref to documentation whenever needed.

3. Execute the programming with hyperparameters set to values provided in the tutorial.

![vanillaRNN_structure](images/2019/05/vanillarnn-structure.png)

In the official tutorial of PyTorch, the `rNN.forward` method only contains on time path, i.e. a slice of sequential input is feed in the neural net.

This might be a good practice since it make the "data flow" inside the neural net more clear. However, I am not sure if this is a good idea when it comes to batch training......

### word-level text generation

### References

- [Medium post](https://medium.com/coinmonks/character-to-character-rnn-with-pytorchs-lstmcell-cd923a6d0e72): Character-To-Character RNN With Pytorch’s LSTMCell by [Ulyanin](https://medium.com/@stepanulyanin)
- Official [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html): Character-level RNN for name classification

## Algorithm

See Onenote Notebook!

## Q & A

#### Pure rNN for image classification?

Stackoverflow problem answered by [@David Parks](https://stackoverflow.com/a/49495862)

> Aymericdamien has some of the best examples out there, and they have an example of using an RNN with images.
> [repo](https://github.com/aymericdamien/TensorFlow-Examples) and [notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb)
>
> The example is using MNIST, but it can be applied to any image.
>
> However, I'll point out that you're unlikely to find many examples of using an RNN to classify an image because RNNs are inferior to CNNs for most image processing tasks. The example linked to above is for educational purposes more than practical purposes.
>
> Now, if you are attempting to use an RNN because you have a sequence of images you wish to process, such as with a video, in this case a more natural approach would be to combine both a CNN (for the image processing part) with an RNN (for the sequence processing part). To do this you would typically pretrain the CNN on some classification task such as Imagenet, then feed the image through the CNN, then the last layer of the CNN would be the input to each timestep of an RNN. You would then let the entire network train with the loss function defined on the RNN.

#### Why programmers prefer ASCII over UTF8?

Stackexchange problem answered by [@Mchl](https://softwareengineering.stackexchange.com/a/97376)

> 81
>
> In some cases it can speed up access to individual characters. Imagine string `str='ABC'` encoded in UTF8 and in ASCII (and assuming that the language/compiler/database knows about encoding)
>
> To access third `c` character from this string using array-access operator which is featured in many programming languages you would do something like `c = str[2]`.
>
> Now, if the string is ASCII encoded, all we need to do is to fetch third byte from the string.
>
> If, however string is UTF-8 encoded, we must first check if first character is a one or two byte char, then we need to perform same check on second character, and only then we can access the third character. The difference in performance will be the bigger, the longer the string.
>
> This is an issue for example in some database engines, where to find a beginning of a column placed 'after' a UTF-8 encoded VARCHAR, database does not only need to check how many characters are there in the VARCHAR field, but also how many bytes each one of them uses.

### Nice Python Tricks

#### Python dir listing

Source: [Eliot's blog](https://www.saltycrane.com/blog/2010/04/options-listing-files-directory-python/)

The general os package (love it!)

```python
import os
dirlist = os.listdir("/usr")

from pprint import pprint
pprint(dirlist)
```

listdir + regex

```python
import os, pprint, re

pat = re.compile(r".+\d.+")
dirlist = filter(pat.match, os.listdir("/usr/local"))

pprint.pprint(dirlist)
```

The python glob approach

```Python
import glob
dirlist = glob.glob('/usr/*')

from pprint import pprint
pprint(dirlist)
```

#### Again, naming convention

From [Stackoverflow](https://stackoverflow.com/a/8423697).
`module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name, local_var_name`

#### See if a python module is imported

```Python
>>> import sys
>>> 'unicodedata' in sys.modules
False
>>> import unicodedata
>>> 'unicodedata' in sys.modules
True
```

#### Pytorch: Stack vs Cat

Stack along a new dimension; Cat along an existing dimension.

```python
import torch
# if s, t of shape [3,4]
torch.cat([s,t]).shape    # gives 6,4
torch.stack([s,t]).shape  # gives 2,3,4
```

#### Pytorch: reshaping tensors

Need to summarize this [page](https://stackoverflow.com/questions/43328632/pytorch-reshape-tensor-dimension)

#### Formating print

```python
>>> num=3.65789
>>> "The number is {:.4f}".format(num)
'The number is 3.6579'
```
