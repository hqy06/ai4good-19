import numpy as np

from sklearn.datasets import load_iris  # for the IRIS data set
from numpy import random  # for data shuffle
from scipy import stats  # for neighbor voting


class kNNIris:
    def __init__(self, k):
        """
        Init the kNN classifier with user-specific hyperpara k
        """
        self.k = k
        self.iris = None
        # self.iris_test_data = None
        # self.iris_test_target = None
        # self.iris_train_data = None
        # self.iris_train_target = None

    def load_data(self):
        """
        Let's use the IRIS dataset via sklearn; shuffled, split into train/test sets with 2:1 ratio
        :return: the shuffled labelled dataset
        """
        # load the data set
        iris = load_iris()

        # parallel shuffle the dataset WITH labels
        data = iris['data']
        target = iris['target']
        num_datapoint = data.shape[0]
        target = target.reshape(num_datapoint, 1)
        labeled_iris = np.hstack([data, target])
        random.shuffle(labeled_iris)

        self.iris = labeled_iris
        return self.iris

    def train_test_split(self, ratio=0.6):
        if ratio <= 0 or ratio >= 1:
            raise AssertionError

        # Train/Test split
        size = self.iris.shape[0]
        train_size = int(size * ratio)
        test_size = size - train_size
        iris_train = self.iris[:train_size]
        iris_test = self.iris[-test_size:]

        # Separate data and targets
        iris_train_data = iris_train[:, 0:4]
        iris_train_target = iris_train[:, 4]
        iris_test_data = iris_test[:, 0:4]
        iris_test_target = iris_test[:, 4]

        return iris_train_data, iris_train_target, iris_test_data, iris_test_target

    def distance(self, data, x):
        """
        calculate the euclidean distance between an array of points towards a given point
        """
        return np.sqrt(np.sum(np.sum((data - x) ** 2, axis=1)))

    def predict(self, x_train, t_train, x_query):
        """
        Given a valid query point x, try to classify it.
        """
        distances = self.distance(x_train, x_query)
        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[:self.k]
        k_modal, k_modal_count = stats.mode(t_train[k_indices])
        return k_modal

    def test(self, x_train, t_train, x_test, t_test, test_number=10):
        """
        Use test_data to evaluate the performance of this classifer.
        """
        rand_test_indices = random.randint(0, x_test.shape[0], test_number)
        count = 0
        for test_index in rand_test_indices:
            prediction = self.predict(x_train, t_train, x_test[test_index])
            actual = t_test[test_index]
            print('Predict: ', prediction[0], " Actually: ", actual)

            if prediction == actual:
                count += 1

        print('Accuracy: ', count, " out of ", test_number)

        return None


def main():
    my_iris = kNNIris(15)
    my_iris.load_data()
    x_train, t_train, x_test, t_test = my_iris.train_test_split()
    my_iris.test(x_train, t_train, x_test, t_test)


if __name__ == '__main__':
    main()
