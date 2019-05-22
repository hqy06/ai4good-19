"""
Playing with MLP classfier, using sklearn package
-------------------------
Tasks:
0. using the model
0. the MinMaxScaler
1. hyperparamter: stochastic learning rate
2. hyperparamter: regularization penalty
3. visualization:

Read the documentation first:
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
"""

# ========== General Usage
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# ========== Silence the Convergence Warning
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


# ========== 1. Learning rate demo
labels = ["constant learning-rate", "constant with momentum", "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum", "inv-scaling with Nesterov's momentumn", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]

hyperparas = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
               'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
               'nesterovs_momentum': False, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
               'nesterovs_momentum': True, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'invscaling',
               'momentum': 0, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
               'nesterovs_momentum': True, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
               'nesterovs_momentum': False, 'learning_rate_init': 0.2},
              {'solver': 'adam', 'learning_rate_init': 0.01}]


def learning_rate_demo():
    # load the toy datasets
    from sklearn import datasets
    iris = datasets.load_iris()
    digits = datasets.load_digits()
    circles = datasets.make_circles(noise=0.3, factor=0.5, random_state=48)
    moons = datasets.make_moons(noise=0.3, random_state=48)

    datasets = [(iris.data, iris.target),
                (digits.data, digits.target), circles, moons]
    datasets_names = ['iris', 'mnist', 'circle', 'moon']

    # prepare the plotting: 2*2=4 sub figures, each of size 15*10
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # For each datasets, train -> test -> plot.
    # the asterisk sign is for variadic argument
    for ax, data, name in zip(axes.ravel(), datasets, datasets_names):
        lr_plot_me(*data, axis=ax, dataset_name=name)

    # Display which line is which
    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    plt.savefig('lr_demo.png')
    plt.show()


def lr_plot_me(X, y, axis, dataset_name):
    print('\n learning on dataset %s' % dataset_name)
    axis.set_title(dataset_name)    # title for the sub-figure
    X = MinMaxScaler().fit_transform(X)  # scale the features

    if dataset_name == "digits":
        max_iter = 15
    else:   # set maxium iterations
        max_iter = 400
    mlps = []   # an empty list for holding mlp instances

    for label, hyperpara in zip(labels, hyperparas):
        print(">> training: %s" % label)
        mlp = MLPClassifier(random_state=0, max_iter=max_iter, **hyperpara)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("     Training set score: %f" % mlp.score(X, y))
        print("     Training set loss: %f" % mlp.loss_)

    for mlp, label, args in zip(mlps, labels, plot_args):
        axis.plot(mlp.loss_curve_, label=label, **args)


# ========== 2. Regularization penalty

# Setting the hyperparamters
alphas = np.logspace(-5, 3, 5)

# The color map

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])


def regularization_demo():
    # Generate toy dataset
    from sklearn import datasets
    moons = datasets.make_moons(noise=0.3, random_state=48)
    circles = datasets.make_circles(noise=0.2, factor=0.5, random_state=52)
    blobs = datasets.make_blobs(centers=2, n_features=2, random_state=37)

    datasets = [moons, circles, blobs]
    datasets_names = ['moon', 'circle', 'two blobs']

    # Init the figures
    figs, axes = plt.subplots(3, 6, figsize=(30, 12))

    assert (len(datasets) == len(datasets_names)), "size mismatch!"

    row = 0
    for (X, y), name in zip(datasets, datasets_names):
        print('\n learning on dataset %s' % name)
        # Prepare the data
        # - scale the features
        X = StandardScaler().fit_transform(X)
        # - Train test split!
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
        # - canvas boundary
        coord1_min, coord1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        coord2_min, coord2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        h = 0.2
        coord1, coord2 = np.meshgrid(
            np.arange(coord1_min, coord1_max, h), np.arange(coord2_min, coord2_max, h))

        # Plot the scaled data at the first slot of the row
        column = 0
        ax = axes[row, column]
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        ax.scatter(X_test[:, 0], X_test[:, 1],
                   c=y_test, cmap=cm_bright, alpha=0.5)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylabel(name, size=15, labelpad=12)
        column += 1  # finish with this slot

        # Init the models
        mlps = []
        for alpha in alphas:
            mlp = MLPClassifier(alpha=alpha, random_state=3)
            mlp.fit(X, y)
            mlps.append(mlp)

        for mlp, alpha in zip(mlps, alphas):
            ax = axes[row, column]
            reg_plot_me(X_train, y_train, X_test, y_test, coord1, coord2,
                        axis=ax, alpha=alpha, classfier=mlp)
            ax.axis('off')
            column += 1

        row += 1

    # Save and show
    plt.savefig('reg_demo.png')
    plt.show()


def reg_plot_me(X_train, y_train, X_test, y_test, coord1, coord2, axis, alpha, classfier):
    # Plot the decision boundary
    if hasattr(classfier, "decision_function"):
        Z = classfier.decision_function(np.c_[coord1.ravel(), coord2.ravel()])
    else:
        Z = classfier.predict_proba(
            np.c_[coord1.ravel(), coord2.ravel()])[:, 1]
    Z = Z.reshape(coord1.shape)
    axis.contourf(coord1, coord2, Z, cmap=cm, alpha=0.7)

    # Plot the scaled data
    axis.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                 cmap=cm_bright, edgecolors='black')
    axis.scatter(X_test[:, 0], X_test[:, 1],
                 c=y_test, cmap=cm_bright, alpha=0.5, edgecolors='black')

    # Title and Accuracy
    title = 'alpha ' + str(alpha)
    axis.set_title(title, size=10)
    score = classfier.score(X_test, y_test)
    text = ('%.2f' % score).lstrip('0')
    axis.text(coord1.max() - 0.3, coord2.min() + 0.3,
              text, size=12, horizontalalignment='right')


# ========== 3. Regularization penalty

def weight_visualization_demo():
    from sklearn import datasets
    # load and preprocess the data
    digits = datasets.load_digits()
    digits.data = MinMaxScaler().fit_transform(digits.data)
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=.4)

    in_dim = digits.data.shape[1]
    in_width = int(in_dim ** (1 / 2.0))
    print('Using built-in MNIST with %d images of size %d \n' %
          (digits.data.shape[0], digits.data.shape[1]))

    # init and train the mlp
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1)
    mlp.fit(X_train, y_train)

    # performance check
    print("\nTraining set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))

    # visualize some of the coefficients
    vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
    fig, axes = plt.subplots(4, 4)
    # print('1st coeff matrix of size',mlp.coefs_[0].shape )
    for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
        # print('> coef of size', coef.shape)
        ax.matshow(coef.reshape(in_width, in_width), cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_yticks(())
        ax.set_xticks(())

    plt.savefig('visual_para.png')
    plt.show()


# ========== Main Entrance


def main(no_task):
    if no_task == 1:
        learning_rate_demo()
    elif no_task == 2:
        regularization_demo()
    elif no_task == 3:
        weight_visualization_demo()
    else:
        raise ValueError("Task number out of boundary")

    return 0


if __name__ == '__main__':
    task = input("key in 1, 2, or 3: ")
    main(int(task))
