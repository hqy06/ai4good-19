"""
Toying with k-Means, both hard and soft.
------------------------------------
TODO:
- implement soft k-mean assignment
------------------------------------
Method:
- Type: unsupervised
- Approach: clustering
- Probability: naive
------------------------------------
Dataset:
- Source: UCI "wine" via sklearn
- # samples = 178
- # features = 13
- Quality: clean, no missing value
------------------------------------
Math work:
- N samples, l features per sample: R^l sample sapce
- K clusters (hyperpara): m_1, ..., m_k in R^
"""

# ==============================================================
from sklearn import datasets  # make use of the sklearn lib
from sklearn.model_selection import train_test_split
import numpy as np


# ============================================================
# Utilities


def load_wine_data():
    print("*** loading data ....... \n")
    # Load the dataset via sklearn, it is a python library
    dataset = datasets.load_wine()

    # Extract info
    data = dataset['data']
    target = dataset['target']
    target.reshape(data.shape[0], 1)
    target_names = dataset['target_names']
    feature_names = dataset['feature_names']

    # Concat data & label and do the shuffle
    # stack arrays column-wise
    labeled_data = np.hstack([data, target.reshape(data.shape[0], 1)])
    # do the shuffle
    np.random.shuffle(labeled_data)

    shuffled_data = labeled_data[:, :data.shape[1]]
    shuffled_labels = labeled_data[:, data.shape[1]]

    print("*** finish loading data.\n")
    return shuffled_data, shuffled_labels, target_names, feature_names


class kMean():
    def __init__(self, n_features, k, n_samples, max_iter=256, tolerance=0.001, is_soft=False):
        self.k = k  # number of clusters
        self.samples = np.zeros((k, n_samples))  # one point per row
        self.centroids = np.zeros((k, n_features))  # one center per row
        self.tolerance = tolerance  # stop iter if the |old-new| bounded by this
        self.max_iter = max_iter  # max iteration time
        self.soft = is_soft  # soft k-mean or not?
        self.trained = False  # has the model been fit?

    def fit(self, train_data, iter=64, random=True, verbose=False):
        """ Fit the model by assign/refit
        @para: train_data  a 2d-array of datapoints to cluster."""
        # Check if the value of iter is valid
        if verbose:
            print('*** fitting the model....\n')
        if iter > self.max_iter or iter < 1:
            raise ValueError('Iteration out of boundary, got ', iter,
                             ' should be an int between 1 and ', self.max_iter)

        # Randomness of initial cluster_centers
        if random is True:
            self.centroids = np.random.uniform(
                low=-1, high=-1, size=(self.k, self.centroids.shape[1]))

        # Run the k_means algorithm
        self.centroids, train_predicts = _k_means(
            self.centroids, train_data, iter, self.tolerance, self.soft)

        self.trained = True
        if verbose:
            print('*** finish fitting model....\n')
        return self.centroids, train_predicts

    def predict(self, xx):
        """Make prediction by assign responsibility to query point"""
        if self.trained is False:
            print('Warning: this model hasn\'t been train yet!')
        distances = _dist(xx, self.centroids)
        k = np.argmax(distances, axis=None)
        return k


def _k_means(init_centroids, train_data, iter, tolerance, isSoft=False):
    # extract key parameters
    n_samples = train_data.shape[0]
    n_clusters = init_centroids.shape[0]
    n_features = train_data.shape[1]

    # sanity checks
    if n_clusters >= n_samples:
        raise ValueError(
            'number of samples should be no smaller than number of clusters')
    if iter <= 1:
        raise ValueError('number of iteration should be non-neg int')

    responsibilities = np.zeros((n_samples, n_clusters))  # one sample per row
    centroids = init_centroids

    # Assign & Refit
    for epoch in range(iter):
        # assign responsibility
        responsibilities = _k_means_assignment(
            train_data, centroids, isSoft=isSoft)

        # if not within tolerance, update; else, just out the loop
        new_centroids = _k_means_refit(
            train_data, responsibilities)
        if np.allclose(centroids, new_centroids, rtol=tolerance):
            break
        else:
            centroids = new_centroids

    return centroids, responsibilities


def _k_means_assignment(samples, centroids, norm='euclidean', isSoft=False):
    n_samples = samples.shape[0]
    n_clusters = centroids.shape[0]
    responsibilities = np.zeros((n_samples, n_clusters))
    for i in range(n_samples):  # iterate through all samples
        sample = samples[i]  # The i-th row in the samples matrix
        distances = _dist(sample, centroids)

        if isSoft:
            raise Exception('soft K-Mean not implemented!')
        else:  # hard k-mean
            j = np.argmax(distances, axis=None)  # range: 0 to n_clusters-1
            # Set the i+1 row of responsibilities matrix to (0...010...0)
            responsibilities[i, :] = np.eye(n_clusters)[j]

    # Sanity check: every row should sum to 1
    if not isSoft and not np.array_equal(responsibilities.sum(axis=1), np.ones(n_samples)):
        raise Exception('something wrong happened in the assignment stage')
    elif isSoft and not np.isclose(responsibilities.sum(axis=1), np.ones(n_samples)):
        raise Exception('something wrong happened in the assignment stage')
    else:
        pass

    return responsibilities


def _k_means_refit(samples, responsibilities):
    n_samples = responsibilities.shape[0]
    n_clusters = responsibilities.shape[1]
    n_features = responsibilities.shape[1]
    centers = np.zeros((n_clusters, n_features))

    for k in range(n_clusters):
        # observe column k in the responsibility matrix
        resp_k = responsibilities[:, k]
        # extract those with nonzero responsibility
        valid_ind = np.nonzero(resp_k)
        valid_resp = resp_k[valid_ind]
        valid_samples = samples[valid_ind]
        n_valid = len(valid_ind)

        # calculation
        # resp is considered as weight
        weighted_samples = (
                valid_resp * valid_samples.reshape(n_features, n_valid)).reshape(n_valid, n_features)
        # sum over samples (axis=0) and divide by n_valid to obtained the (weighted) avg
        center_k = np.mean(weighted_samples, axis=0)

        # write into the centroid matrix
        centers[k, :] = center_k

    return centers


def _dist(x, yy, norm='euclidean'):
    # Here x is a point of shape (d,)
    # and yy is a set points of hstacked as (N,d)
    # the result would be a (N,) array
    if norm is not 'euclidean':
        raise Exception('currently L2 norm only!')

    return np.linalg.norm(yy - x, ord=2, axis=1)


# =====================================================================


def main():
    # 1. Load the UCI "wine" data, no ploting...
    data, labels, label_names, feature_names = load_wine_data()
    n_features = data.shape[1]
    n_labels = label_names.shape[0]
    n_samples = data.shape[0]
    print('number of feature = ', n_features,
          '\nnumber of labels = ', n_labels,
          '\nnumber of samples = ', n_samples, end='\n\n')

    # 2. Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.33)

    # 3. Init a kMean
    kmean = kMean(n_features, n_labels, n_samples, is_soft=False)

    # 4. Train the classifier
    kmean.fit(X_train)

    # 5. Performance on training set
    print('Label | Predic \n')
    for rand_index in np.random.randomin(0, X_train.shape[0], 10):
        x = X_train[rand_index]
        y = y_train[rand_index]
        pred = kmean.predict(x)
        print('%d   |  %d \n', y, pred)

    # 6. Performance on test set
    print('Label | Predic \n', y, pred)
    for rand_index in np.random.randomin(0, X_test.shape[0], 10):
        x = X_test[rand_index]
        y = y_test[rand_index]
        pred = kmean.predict(x)
        print('%d   |  %d \n', y, pred)


if __name__ == "__main__":
    # execute only if run as a script
    main()
