"""
helper functions
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

'''remove annoying DeprecationWarning from sklearn script'''
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

LOGGER = logging.getLogger(os.path.basename(__file__))


def write_results(y_pred, y_true):
    """
    writes a confusion matrix, etc
    """
    pass


def plot_results(train_acc, test_acc, param_pairs, exp_name):
    param_pairs = [str(param) for param in param_pairs]
    plt.rcParams.update({'font.size': 6})

    x = np.arange(len(param_pairs))
    plt.plot(x, train_acc)
    plt.plot(x, test_acc)
    plt.xticks(x, param_pairs)
    plt.ylim(0., 1.)
    plt.legend(['train accuracy', 'test accuracy'])
    plt.xlabel('parameter pairs')
    plt.ylabel(yaxis)
    plt.savefig('./figures/{}.png'.format(exp_name))
    plt.close()
    #plt.clf()


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
        all_features = list(range(X_train.shape[1]))
        keep_features = [x for x in all_features if x not in [2, 6, 7]]
        X_train = X_train[:, keep_features]
        X_test  = X_test[:, keep_features]

    # test_mode uses a small subset of the data
    if test_mode:
        LOGGER.info('running in test mode, train n=500')
        X_train = X_train[:500, :]
        y_train = y_train[:500]

    # data is accessed as data['X']['test']
    data = {
        'X': {'train': X_train, 'test': X_test},
        'y': {'train': y_train, 'test': y_test}
    }

    LOGGER.debug('n TRAIN = {}, n TEST = {}'.format(
        X_train.shape[0], X_test.shape[0]))

    return(data)


def load_covertype(test_mode=False, test_pct=0.1):
    ## TODO: we're going to scale all of the one-hot encoded features later...
    ##       this is likely bad (but for now, it's fine)... jdv
    ##       this is a good solution:
    ##       https://scikit-learn.org/dev/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py

    data = np.genfromtxt('data/covtype.csv', delimiter=',')

    # shuffle data as we extract it to X and y
    idx = np.arange(len(data))
    np.random.shuffle(idx)

    X = data[idx, :-1]
    le = LabelEncoder()
    y = le.fit_transform(data[idx, -1])

    # split into training and test set
    n_test = int(np.floor(len(X) * test_pct))

    X_test = X[:n_test, :]
    y_test = y[:n_test]
    X_train = X[n_test:, :]
    y_train = y[n_test:]

    # test_mode uses a small subset of the data
    if test_mode:
        LOGGER.info('running in test mode, train n=5000')
        X_train = X_train[:5000, :]
        y_train = y_train[:5000]

    # data is accessed as data['X']['test']
    data = {
        'X': {'train': X_train, 'test': X_test},
        'y': {'train': y_train, 'test': y_test}
    }

    LOGGER.debug('n TRAIN = {}, n TEST = {}'.format(
        X_train.shape[0], X_test.shape[0]))

    return(data)

