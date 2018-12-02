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
    model.fit(data['X']['train'], data['y']['train'])
    y_train_pred = model.predict(data['X']['train'])   # train scores
    y_test_pred =  model.predict(data['X']['test'])    # test scores

    model_train_acc = accuracy_score(y_train_pred, data['y']['train'])
    model_test_acc = accuracy_score(y_test_pred, data['y']['test'])

    LOGGER.info('train/valid accuracy: {}/{}'.format(
        model_train_acc, model_test_acc))

    results = {'train': model_train_acc, 'test':  model_test_acc}

    return results, model

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
    decision_tree = models.decision_tree()
    _, single_model = kfold_train_loop(data, decision_tree)
    estimator = single_model.best_estimator_.named_steps['clf']
    
    storage = {'train_acc': [], 'test_acc': []}
    for max_depth, n_learners in param_pairs:
        model = models.random_forest(estimator, max_depth=max_depth, n_learners=n_learners)
        results, _ = kfold_train_loop(data, model)
        storage['train_acc'].append(results['train'])
        storage['test_acc'].append(results['test'])

    utils.plot_decision_tree_result(
        storage['train_acc'], storage['test_acc'], param_pairs)

    return results


def boosted_nn_baseline(data):
    """baseline -- neural network with adaboost"""
    # get the non-boosted model results
    model = models.mlp()
    single_results, single_best_model = kfold_train_loop(data, model)

    # get the boosted model results
    model = model.mlp_boosted(single_best_model)
    boosted_results, boosted_best_model = kfold_train_loop(data, model)

    return(single_results, boosted_results, single_best_model, boosted_best_model)

