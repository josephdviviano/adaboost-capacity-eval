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
'''remove annoying DeprecationWarning from sklearn script'''

CORR_FEATURES = [2, 6, 7]
LOGGER = logging.getLogger(os.path.basename(__file__))

def write_results(y_pred, y_true):
    """
    writes a confusion matrix, etc
    """
    pass


def _relabel_wine(targets, num_classes=5):
    """
    Reduce the number of label by merging
    some classes together.

    NOTE: The number of classes for the
    wine dataset is 7, but theres is too
    few examples for class 3 and 9. It
    Creates bugs when k-folding
    """
    if num_classes == 'all':
        pass
    elif num_classes == 5:
        targets[targets==3.] = 4.
        targets[targets==9.] = 8.
    elif num_classes == 3:
        targets[targets in [1., 2., 3.]] = 1.
        targets[targets in [4., 5., 6.]] = 2.
        targets[targets in [7., 8., 9.]] = 3.
    else:
        raise ValueError('Valid value for num_classes are: 5, 7')
    return(targets)


def load_wine(test_mode=False, remove_corr_features=False):
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
    y_train = _relabel_wine(data_train[:, -1])
    y_test  = _relabel_wine(data_test[:, -1])

    if remove_corr_features:
        all_features = list(range(X_train.shape[1]+1))
        keep_features = [x for x in all_features if x not in CORR_FEATURES]
        X_train = data_train[:, keep_features]
        X_test  = data_test[:, keep_features]

    # test_mode uses a small subset of the data
    if test_mode:
        LOGGER.info('running in test mode, n=500')
        n_samples = 500
    else:
        n_samples = len(X_train)

    # data is accessed as data['X']['test']
    data = {
        'X': {'train': X_train, 'test': X_test},
        'y': {'train': y_train, 'test': y_test}
    }

    LOGGER.debug('n TRAIN = {}, n TEST = {}'.format(
        X_train.shape[0], X_test.shape[0]))

    return(data)


def plot_decision_tree_result(train_acc, test_acc, params_pairs):
    params_pairs = [str(param) for param in params_pairs]
    plt.rcParams.update({'font.size': 6})
    plt.plot(params_pairs, train_acc)
    plt.plot(params_pairs, test_acc)
    plt.legend(['train accuracy', 'test accuracy'])
    plt.xlabel('parameter pairs')
    plt.ylabel('gradient')
    plt.savefig('./figures/decision_tree.png')
    plt.show()


def load_covertype(test_mode=False, valid_pct=0.1):
    pass


