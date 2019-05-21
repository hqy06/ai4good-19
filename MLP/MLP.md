## Multi-Layer Percptron

## Summary

- Just a basic feed-forward neural network, _fully_ connected.
- It can do the supervised learning as well as the unsupervised one.
- Motivation: neurons & function approximation
- Some Limitations
  - Too may parameters...
  - Prone to overfit (so lets bring in regularization!)
  - Cannot do parallel computation
- Training: use the backpropagation approach, which is merely multivariate chain rule of (partial) derivtives.

### Algorithm

Nothing Fancy, check study note

### Implementation

#### Hints

- For dot product $x \cdot y$: `x.dot(y)`
- For matrix transpose $ x^\top$ use `x.transpose`
- Keep track of the dimensions of all numpy arrays
- Start with a simpleMLP, then to a standard one.
-

#### Coding

A not-running undebugged version can be find in simpleMLP.py. The file is for practice algorithm implementation only and, unlike k-mean and k-NN, is not supposed to be a functional code.

### Using the Libraries

#### Sci-Kit Learn

Just like linear transformation (not linear functional!) and manifold, the MLP can be view as a mathematical model that can be used in the field of ML. So there is no supersing that `sklearn` got their MLP prepared for both classification and regression.

Link [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network) :no_mouth:

**MLP regression**

TBA

**MLP classification**

TBA

#### PyTorch

This PyTorch, what would you expect?

...Well, you just instantiate the model and play with datasets and hyperparmeters

### Reflections

### Note to Self

1. sklearn: `make_moon` and `make_circle`
   > make_circles and make_moons generate 2d binary classification datasets that are challenging to certain algorithms (e.g. centroid-based clustering or linear classification), including optional Gaussian noise. They are useful for visualisation. make_circles produces Gaussian data with a spherical decision boundary for binary classification, while make_moons produces two interleaving half circles.
2. MinMaxScaler vs. StandardScaler: [link1](https://www.quora.com/Minmaxscaler-vs-Standardscaler-Are-there-any-specific-rules-to-use-one-over-the-other-for-a-particular-application), [link2](http://rajeshmahajan.com/standard-scaler-v-min-max-scaler-machine-learning/)

3.matplotlib tricks

- To hide axis value, do this: `ax.set_yticklabels([])`
- To add some fancy text in the subplot, do this: `axis.text(pos_x, pos_y,text, size=12, horizontalalignment='right')`
- To set some fashionable (blah) labels, use this: `ax.set_ylabel(name,size=15,labelpad=12)`
