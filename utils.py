"""
helper functions
"""

from sklearn.preprocessing import LabelEncoder
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

LOGGER = logging.getLogger(os.path.basename(__file__))

def get_y_map(data):
    """gets all the unique values in y to allow str <-> int conversion"""
    y_map = LabelEncoder()
    y_map.fit(data['y']['train'])
    return(y_map)


def convert_y(y, y_map):
    """converts all y in data to int if str, else str if int"""
    # convert integers to string labels
    if np.issubdtype(type(y[0]), np.number):
        return(y_map.inverse_transform(y))

    # convert string labels to integers
    else:
        return(y_map.transform(y))


def write_results(y_pred, y_true):
    """writes a confusion matrix, etc"""
    pass


def load_wine(test_mode=False, valid_pct=0.1):
    """
    loads the data into a structure for SCIKIT LEARN. data is stored as
    (n_subjects x n_features).
    """
    X_train = np.load('data/train_images.npy', encoding='latin1')
    X_test  = np.load('data/test_images.npy', encoding='latin1')
    y_train = np.genfromtxt('data/train_labels.csv', names=True, delimiter=',',
        dtype=[('Id', 'i8'), ('Category', 'S20')])
    y_test  = np.genfromtxt('data/train_labels.csv', names=True, delimiter=',',
        dtype=[('Id', 'i8'), ('Category', 'S20')])

    # get data into numpy matrices

    # test_mode uses a small subset of the data
    if test_mode:
        LOGGER.info('running in test mode, n=500')
        n_samples = 500
    else:
        n_samples = len(X_train)

    # make validation set
    n_valid = int(np.floor(valid_pct * n_samples))

    X_valid = X_train[:n_valid, :]
    X_train = X_train[n_valid:, :]
    y_valid = y_train[:n_valid]
    y_train = y_train[n_valid:]

    # data is accessed as data['X']['valid']
    data = {'X': {'train': X_train, 'valid': X_valid, 'test': X_test},
            'y': {'train': y_train, 'valid': y_valid, 'test': y_test}
    }

    LOGGER.debug('n TRAIN = {}, n VALID = {}, n TEST = {}'.format(
        X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

    return(data)



def load_covertype(test_mode=False, valid_pct=0.1):
    pass


