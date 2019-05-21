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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# ========== Silence the Convergence Warning
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# ========== Global Variables
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


# ========== 1. Learning rate demo

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


def main(no_task):
    if no_task == 1:
        learning_rate_demo()
    else:
        raise ValueError("Task number out of boundary")

    return 0


if __name__ == '__main__':
    task = input("key in 1, 2, or 3: ")
    main(int(task))
