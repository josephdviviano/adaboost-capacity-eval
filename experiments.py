"""
holds different experiment functions (import and run these in train.py)
"""
from copy import copy
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
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
    t1 = time.time()
    model.fit(data['X']['train'], data['y']['train'])
    y_train_pred = model.predict(data['X']['train'])   # train scores
    y_test_pred =  model.predict(data['X']['test'])    # test scores

    model_train_acc = accuracy_score(y_train_pred, data['y']['train'])
    model_test_acc = accuracy_score(y_test_pred, data['y']['test'])
    model_train_f1 = f1_score(y_train_pred, data['y']['train'], average='macro')
    model_test_f1 = f1_score(y_test_pred, data['y']['test'], average='macro')

    t2 = time.time()
    LOGGER.info('train/test accuracy after cross-val in {} sec: {}/{}'.format(
        t2-t1, model_train_acc, model_test_acc))

    results = {'train': {'accuracy': model_train_acc, 'f1': model_train_f1},
               'test':  {'accuracy': model_test_acc,  'f1': model_test_f1}
    }

    return(results, model)


def svm(data, n_estimators, experiment_name, boosted=False):
    """linear svm with and without adaboost"""
    # get the non-boosted model results
    model = models.SVM()
    _, single_best_model = kfold_train_loop(data, model)
    estimator = single_best_model.best_estimator_.named_steps['clf']

    # use optimal parameter C to generate param_pairs
    C = estimator.C
    param_pairs = [(C/n, n) for n in n_estimators] if boosted else [(C/n, 1) for n in n_estimators] 

    storage = {'train_acc': [], 'test_acc': [], 'train_f1': [], 'test_f1': []}
    for C, n_learners in param_pairs:
        model = models.boosted_SVM(estimator, C=C, n_learners=n_learners)
        results, _ = kfold_train_loop(data, model)
        storage['train_acc'].append(results['train']['accuracy'])
        storage['test_acc'].append(results['test']['accuracy'])
        storage['train_f1'].append(results['train']['f1'])
        storage['test_f1'].append(results['test']['f1'])

    utils.plot_results(
        storage['train_acc'], storage['test_acc'], param_pairs,
        '{}_accuracy'.format(experiment_name), 'Accuracy'
    )

    utils.plot_results(
        storage['train_f1'], storage['test_f1'], param_pairs,
        '{}_f1'.format(experiment_name), 'F1'
    )

    return(storage)


def decision_tree(data, n_estimators, experiment_name, boosted=False):
    """decision tree with and without adaboost"""
    decision_tree = models.decision_tree(data)
    _, single_model = kfold_train_loop(data, decision_tree)
    estimator = single_model.best_estimator_.named_steps['clf']

    LOGGER.info('**Best parameters**')
    LOGGER.info('max_depth: {}'.format(estimator.max_depth))
    LOGGER.info('min_samples_split: {}'.format(estimator.min_samples_split))
    LOGGER.info('min_samples_leaf: {}'.format(estimator.min_samples_leaf))
    LOGGER.info('max_features: {}'.format(estimator.max_features))
    LOGGER.info('min_impurity_decrease: {}'.format(estimator.min_impurity_decrease))

    init_max_depth = estimator.max_depth
    max_depths = list(range(init_max_depth+len(n_estimators), init_max_depth, -1))
    param_pairs = list(zip(max_depths, n_estimators)) if boosted else list(zip(max_depths, [1]*len(n_estimators)))

    storage = {'train_acc': [], 'test_acc': [], 'train_f1': [], 'test_f1': []}
    for max_depth, n_learners in param_pairs:
        model = models.random_forest(estimator, max_depth=max_depth, n_learners=n_learners)
        results, _ = kfold_train_loop(data, model)
        storage['train_acc'].append(results['train']['accuracy'])
        storage['test_acc'].append(results['test']['accuracy'])
        storage['train_f1'].append(results['train']['f1'])
        storage['test_f1'].append(results['test']['f1'])

    utils.plot_results(
        storage['train_acc'], storage['test_acc'], param_pairs,
        '{}_accuracy'.format(experiment_name), 'Accuracy'
    )

    experiment_name = (
        '{}-{}'.format(experiment_name, 'boosted') if boosted else '{}-{}'.format(experiment_name, 'not-boosted')
    )
    utils.plot_results(
        storage['train_f1'], storage['test_f1'], param_pairs,
        '{}_f1'.format(experiment_name), 'F1'
    )

    return(storage)


def mlp(data, n_estimators, experiment_name, boosted=False):
    """mlp with and without adaboost"""
    # get the non-boosted model results
    model = models.mlp()
    _, single_model = kfold_train_loop(data, model)
    estimator = single_model.best_estimator_.named_steps['clf']

    # use optimal parameter C to generate param_pairs
    hidden_size = estimator.hidden_layer_sizes
    hidden_sizes = [int(np.floor(hidden_size / n)) for n in n_estimators]
    param_pairs = list(zip(hidden_sizes, n_estimators)) if boosted else list(zip(hidden_sizes, [1] * len(n_estimators))) 

    storage = {'train_acc': [], 'test_acc': [], 'train_f1': [], 'test_f1': []}
    for n_hid, n_learners in param_pairs:
        model = models.boosted_mlp(estimator, n_hid=n_hid, n_learners=n_learners)
        boosted_results, boosted_best_model = kfold_train_loop(data, model)
        storage['train_acc'].append(results['train']['accuracy'])
        storage['test_acc'].append(results['test']['accuracy'])
        storage['train_f1'].append(results['train']['f1'])
        storage['test_f1'].append(results['test']['f1'])

    utils.plot_results(
        storage['train_acc'], storage['test_acc'], param_pairs,
        '{}_accuracy'.format(experiment_name), 'Accuracy'
    )

    experiment_name = '{}-{}'.format(experiment_name, 'boosted') if boosted else experiment_name
    utils.plot_results(
        storage['train_f1'], storage['test_f1'], param_pairs,
        '{}_f1'.format(experiment_name), 'F1'
    )

    return(storage)


