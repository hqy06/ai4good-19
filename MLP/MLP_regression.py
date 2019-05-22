"""
Playing with MLP regressor.
------------------------------------
Tasks:
0. implement the stuff
1. generate or find good datasets
2. more on matplotlib
"""

# ========== Import general stuffs
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from matplotlib import cm
from sklearn.neural_network import MLPRegressor
from mpl_toolkits.mplot3d import Axes3D


# # ========== Datasets
# """ No we are not using these any more as they are hard to visualize
# - linnerud
# 20 samples, 3-dim input (int), 3-dim output (int)
# - friedman
# 100 sampels, 4-dim input (float64), 1-dim output (float64)
# - diabetes
# 422 samples, 4-dim input (float64), 1-dim output (float64)"""
# linnerud = sklearn.datasets.load_linnerud()     # multivariate regression
# diabetes = sklearn.datasets.load_diabetes()     # regression
# friedman = sklearn.datasets.make_friedman3()    # arctan transformation
#
# datasets = [(linnerud.data, linnerud.target),
#             (diabetes.data, diabetes.friedman),
#             friedman]
# datasets_names = ['linnerud', 'diabetes', 'friedman 3' ]


# ========== Main Entrance
def main():
    # First we generate a parabolid-like dataset
    N_SAMPLES, A, B = 100, 3., 5.
    X = np.random.uniform(low=-10, high=10, size=(N_SAMPLES, 2))
    f = np.sqrt((X[:, 0] ** 2) / (A ** 2) + (X[:, 1] ** 2) / (B ** 2))
    y = f + 0.5 * np.random.randn(N_SAMPLES)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    # Init the regressor
    mlp = MLPRegressor(activation='relu', max_iter=1000,
                       learning_rate_init=0.005)

    # Train the regressor
    print('Training the regressor...', end='\n')
    mlp.fit(X_train, y_train)
    print("     Training set score: %f" % mlp.score(X_train, y_train))
    print("     Testing set score: %f" % mlp.score(X_test, y_test))

    # Do some test
    for index in np.random.randint(0, y_test.shape[0], 10):
        actual = y_test[index]
        data = X_test[index, :].reshape(1, -1)
        prediction = mlp.predict(data)
        print('actual=%.3f, prediction=%.3f' % (actual, prediction))

    # Plot me!
    # - the canvas
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # - plot the prediction (surface)
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    x_min = min(x1_min, x2_min)
    x_max = max(x1_max, x2_max)
    xx = np.arange(x_min, x_max, 0.25)
    xx1, xx2 = np.meshgrid(xx, xx)
    XX = np.vstack((xx1.ravel(), xx2.ravel())).T
    pred = mlp.predict(XX)
    surf = ax.plot_trisurf(xx1.ravel(), xx2.ravel(), pred,
                           linewidth=0, antialiased=False, alpha=0.1, cmap='viridis', edgecolor='none')

    # - the equation that generates the dataset
    ZZ = np.sqrt((xx1 ** 2) / (A ** 2) + (xx2 ** 2) / (B ** 2))
    ssurf = ax.plot_surface(xx1, xx2, ZZ, linewidth=1, antialiased=False,
                            alpha=0.1, cmap='plasma', edgecolor='none')

    # - the original data (scatter plot)
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train,
               c=y_train, marker='o', cmap='Blues', alpha=1.0)
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c=y_test,
               marker='^', cmap='Blues', alpha=1.0)

    # - deal with labels
    ax.set_yticks(())
    ax.set_xticks(())
    ax.set_zticks(())
    ax.set_title('MLP Regressor Demo', size=12)
    fig.colorbar(surf, shrink=1, aspect=5)
    fig.tight_layout()
    plt.savefig('regressor_demo.png')
    plt.show()
    return 0


if __name__ == '__main__':
    main()
