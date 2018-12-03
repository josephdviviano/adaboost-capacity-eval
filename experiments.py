"""
holds different experiment functions (import and run these in train.py)
"""
from copy import copy
from scipy import stats
from sklearn.metrics import accuracy_score
import logging
import matplotlib.pyplot as plt
import models
import numpy as np
import os
import time
import utils

LOGGER = logging.getLogger(os.path.basename(__file__))


def kfold_train_loop(data, model):
    """
    trains a model using stratified kfold cross validation for hyperparameter
    selection, which is expected to be performed inside the submitted model as
    part of the pipeline.
    """
    model.fit(data['X']['train'], data['y']['train'])
    y_train_pred = model.predict(data['X']['train'])   # train scores
    y_test_pred =  model.predict(data['X']['test'])    # test scores

    model_train_acc = accuracy_score(y_train_pred, data['y']['train'])
    model_test_acc = accuracy_score(y_test_pred, data['y']['test'])

    LOGGER.info('train/valid accuracy: {}/{}'.format(
        model_train_acc, model_test_acc))

    results = {'train': model_train_acc, 'test':  model_test_acc}

    return(results, model)


def svm(data):
    """linear svm with and without adaboost"""
    # get the non-boosted model results
    model = models.SVM()
    single_results, single_best_model = kfold_train_loop(data, model)

    # get the boosted model results
    model = models.boosted_SVM(single_best_model) # returns a model ready to train
    results, best_model = kfold_train_loop(data, model)
    return(results, best_model)


def decision_tree(data, param_pairs):
    """
    Decision tree experiment
    """
    decision_tree = models.decision_tree(data)
    _, single_model = kfold_train_loop(data, decision_tree)
    estimator = single_model.best_estimator_.named_steps['clf']

    LOGGER.info('**Best parameters**')
    LOGGER.info('max_depth: {}'.format(estimator.max_depth))
    LOGGER.info('min_samples_split: {}'.format(estimator.min_samples_split))
    LOGGER.info('min_samples_leaf: {}'.format(estimator.min_samples_leaf))
    LOGGER.info('max_features: {}'.format(estimator.max_features))
    LOGGER.info('min_impurity_decrease: {}'.format(estimator.min_impurity_decrease))

    storage = {'train_acc': [], 'test_acc': []}
    for max_depth, n_learners in param_pairs:
        model = models.random_forest(estimator, max_depth=max_depth, n_learners=n_learners)
        results, _ = kfold_train_loop(data, model)
        storage['train_acc'].append(results['train'])
        storage['test_acc'].append(results['test'])

    utils.plot_results(
        storage['train_acc'], storage['test_acc'], param_pairs, experiment_name)

    return(storage)


def mlp(data, param_pairs, experiment_name):
    """mlp with and without adaboost"""
    # get the non-boosted model results
    model = models.mlp(n_hid=param_pairs[0][0])
    single_results, single_model = kfold_train_loop(data, model)

    storage = {'train_acc': [], 'test_acc': []}

    # get the boosted model results using learned single model hyperparamaters
    for n_hid, n_learners in param_pairs:
        model = models.boosted_mlp(single_model, n_hid=n_hid, n_learners=n_learners)
        boosted_results, boosted_best_model = kfold_train_loop(data, model)
        storage['train_acc'].append(boosted_results['train'])
        storage['test_acc'].append(boosted_results['test'])

    utils.plot_results(
        storage['train_acc'], storage['test_acc'], param_pairs, experiment_name)

    return(storage)


