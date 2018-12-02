"""
holds different experiment functions (import and run these in train.py)
"""
from copy import copy
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import logging
import matplotlib.pyplot as plt
import models
import numpy as np
import os
import time
import utils

LOGGER = logging.getLogger(os.path.basename(__file__))

SETTINGS = {
    'folds': 5,
}


def kfold_train_loop(data, model):
    """
    trains a model using stratified kfold cross validation. hyperparameter
    selection is expected to be performed inside the submitted model as part
    of the pipeline.
    """
    X_train = data['X']['train']
    y_train = data['y']['train']

    kf = StratifiedKFold(n_splits=SETTINGS['folds'], shuffle=True)

    best_model_acc = -1
    last_time = time.time()

    for i, (train_idx, test_idx) in enumerate(kf.split(X_train, y_train)):

        this_time = time.time()
        LOGGER.info("fold {}/{}, {:.2f} sec elapsed".format(
            i+1, SETTINGS['folds'], this_time - last_time))
        last_time = this_time

        # split training and test sets
        X_fold_train = X_train[train_idx]
        X_fold_test  = X_train[test_idx]
        y_fold_train = y_train[train_idx]
        y_fold_test  = y_train[test_idx]

        # fit model on fold (does all hyperparameter selection ox X_fold_train)
        model.fit(X_fold_train, y_fold_train)
        this_model_predictions = model.predict(X_fold_test)

        this_model_acc = accuracy_score(this_model_predictions, y_fold_test)

        if this_model_acc > best_model_acc:
            best_model = copy(model)

    best_model.fit(data['X']['train'], data['y']['train']) # fit training data
    y_train_pred = best_model.predict(data['X']['train'])  # train scores
    y_valid_pred = best_model.predict(data['X']['valid'])  # validation scores
    y_test_pred = best_model.predict(data['X']['test'])    # test scores

    LOGGER.info('train/valid accuracy: {}/{}'.format(
        accuracy_score(y_train_pred, data['y']['train']),
        accuracy_score(y_valid_pred, data['y']['valid'])
    ))

    results = {
        'train': accuracy_score(y_train_pred, data['y']['train']),
        'valid': accuracy_score(y_valid_pred, data['y']['valid']),
        'test': y_test_pred
    }

    return results, best_model


def svm_nonlinear(data):
    """baseline: SVM (with Kernel)"""
    model = models.SVM_nonlinear(data) # returns a model ready to train
    results, best_model = kfold_train_loop(data, model)
    return(results, best_model)



def boosted_svm_baseline(data):
    """baseline: SVM (without Kernel)"""
    model = models.boosted_SVM(data) # returns a model ready to train
    results, best_model = kfold_train_loop(data, model)
    return results, best_model


def decision_tree(data, param_pairs):
    """
    Decision tree experiment
    """
    storage = {'train_acc': [], 'valid_acc': []}
    for max_depth, n_learners in param_pairs:
        model = models.decision_tree(adaboost=True, max_depth=max_depth, n_learners=n_learners)
        results, best_model = kfold_train_loop(data, model)
        storage['train_acc'].append(results['train'])
        storage['valid_acc'].append(results['valid'])

    utils.plot_decision_tree_result(
        storage['train_acc'], storage['valid_acc'], param_pairs)

    return results, best_model


def nn(data):
    """neural network with and without adaboost"""
    # get the non-boosted model results
    model = models.mlp()
    single_results, single_best_model = kfold_train_loop(data, model)

    # get the boosted model results using the hyperparameters learned on a
    # single model
    model = model.mlp_boosted(single_best_model)
    boosted_results, boosted_best_model = kfold_train_loop(data, model)

    return(single_results, boosted_results, single_best_model, boosted_best_model)

