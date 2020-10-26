"""Utility function to print the confusion matrix"""

import numpy as np


def confusion_matrix(true, pred):
    """ Computes a confusion matrix """
    K = len(np.unique(true))  # Number of classes
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result


def train_test_split(X, y, test_size):
    """ Training and testing samples distribution """
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, test_size)

    X_train = X[split]
    y_train = y[split]
    X_test = X[~split]
    y_test = y[~split]

    return X_train, X_test, y_train, y_test


def accuracy_score(y_true, y_predictions):
    """ Calculates the accuracy score for the model"""
    correct = 0
    for yt, yp in zip(y_true, y_predictions):
        if yt == yp:
            correct += 1

    return correct / len(y_predictions)
