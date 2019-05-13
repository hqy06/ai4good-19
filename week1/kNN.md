## k-Nearest Neighbor

### Note to Self

- Supervised Learning
- Motivation: a generalization of NN, aka Nearest Neighbor
- Limitation
  - There is actually no "training" process. For each quiry, an entire computation process is required. This is not good enough
  - If the training data points are actually separated, the voting for kNN is not convincing as the clusters maybe far away (recall the curse of dimensionality. In this case, let's do SVM and similar.
- Comparing to PCA, aka Principle Component Analysis
  - What PCA is trying to do is to present data sets as linear (maybe no-linear in the future) combinations of a small sets of eigenvectors. This results in a homomorphism(?) that projects into a lower dimensional subsapce.
  - kNN sort of like let-the-centers-of-nearby-clusters-vote.
  - Another perspective: PCA wants fewers features, kNN wants fewer data points
  - Also, kNN can be viewed as a special case of GMM, aka Gaussian Mixture Model

### The Algorithm

```
    1. Load data - create labelled data sets for training/testing/validating
    2. Fit/Train - evaluate norm between query point and points in the training set
    3. Predict - let the nearest k neighbors vote
```

### Application: Fisher's Iris

#### Facts

1. In `sklearn`, the function `load_iris()` would return a dictionary.
2. The original Iris is provided in CVS
3. Recall all the handy trick on _list slicing_.
4. Recall the fancy things we can do with the n-dimensional arrays in numpy.
5. The function `numpy.random()` can be used for shuffling numpy arrays.
6. The voting process can be implemented using `scipy.stats.mode()`, which return an array of modals.

#### Coding

see the ``.py` file

### Good to read

[x] amoeba, [What is the relation between k-means clustering and PCA?](https://stats.stackexchange.com/q/187089) version: 2016-02-11

[x] Ding & He, [K-means Clustering via Principal Component Analysis](http://ranger.uta.edu/~chqding/papers/KmeansPCA1.pdf)
