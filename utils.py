"""
helper functions
"""

from sklearn.preprocessing import LabelEncoder
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
'''remove annoying DeprecationWarning from sk-learn script
'''

LOGGER = logging.getLogger(os.path.basename(__file__))

def write_results(y_pred, y_true):
    """
    writes a confusion matrix, etc
    """
    pass

def load_wine(test_mode=False, valid_pct=0.1):
    """
    loads the data into a structure for SCIKIT LEARN. data is stored as
    (n_subjects x n_features).
    """
    #load data
    data_train = np.genfromtxt('data/wine/train.csv', delimiter=',', skip_header=1)
    data_test = np.genfromtxt('data/wine/test.csv', delimiter=',', skip_header=1)
    
    #split into X and y
    X_train = data_train[:, :-1]
    X_test  = data_test[:, :-1]
    y_train = data_train[:, -1]
    y_test  = data_train[:, -1]

    # to few examples for these two the
    # classes 3 and 9. Creates bugs when 
    # k-folding
    y_train[y_train==3.] = 4.
    y_train[y_train==9.] = 8.
    y_test[y_train==3.] = 4.
    y_test[y_train==9.] = 8.

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
    data = {
        'X': {'train': X_train, 'valid': X_valid, 'test': X_test},
        'y': {'train': y_train, 'valid': y_valid, 'test': y_test}
    }

    LOGGER.debug('n TRAIN = {}, n VALID = {}, n TEST = {}'.format(
        X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

    return data

def load_covertype(test_mode=False, valid_pct=0.1):
    pass


