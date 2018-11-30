"""
helper functions
"""

from sklearn.preprocessing import LabelEncoder
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

LOGGER = logging.getLogger(os.path.basename(__file__))

def get_y_map(data):
    """gets all the unique values in y to allow str <-> int conversion"""
    y_map = LabelEncoder()
    y_map.fit(data)
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


def load_wine(test_mode=False, valid_pct=0.1, test_pct=0.1):
    """
    loads the data into a structure for SCIKIT LEARN. data is stored as
    (n_subjects x n_features).
    """
    data = []
    data_white = csv.reader(open('data/winequality-white.csv',"rt"), delimiter=';')
    next(data_white)
    for row in data_white:
        data.append(list(row))
    data_red   = csv.reader(open('data/winequality-red.csv',"rt"), delimiter=';')
    next(data_red)
    for row in data_red:
        data.append(list(row))

    data= np.array(data)
    np.random.shuffle(data)
    
    X_train = data[:int((1-test_pct-valid_pct)*data.shape[0]),:-1].astype(float)
    X_valid = data[int((1-valid_pct-test_pct)*data.shape[0]):int((1-test_pct)*data.shape[0]),:-1].astype(float)
    X_test  = data[int((1-test_pct)*data.shape[0]):,:-1].astype(float)
    
    
    y_train = data[:int((1-test_pct-valid_pct)*data.shape[0]),-1]
    y_valid = data[int((1-valid_pct-test_pct)*data.shape[0]):int((1-test_pct)*data.shape[0]),-1]
    y_test  = data[int((1-test_pct)*data.shape[0]):,-1]
    
    # data is accessed as data['X']['valid']
    data = {'X': {'train': X_train, 'valid': X_valid, 'test': X_test},
            'y': {'train': y_train, 'valid': y_valid, 'test': y_test}
    }

    LOGGER.debug('n TRAIN = {}, n VALID = {}, n TEST = {}'.format(
        X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

    return(data)



def load_covertype(test_mode=False, valid_pct=0.1):
    pass


