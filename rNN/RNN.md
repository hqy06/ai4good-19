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
