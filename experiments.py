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

def logistic_regression(data, n_estimators, experiment_name, estimator=None, boosted=False):
    """Logistic regression with and without adaboost"""
    # get the non-boosted model results
    if not estimator:
        model = models.logistic_regression(data)
        _, single_best_model = kfold_train_loop(data, model)
        estimator = single_best_model.best_estimator_.named_steps['clf']

    param_pairs = [((n if boosted else 1)) for n in n_estimators] 

    storage = {'train_acc': [], 'test_acc': [], 'train_f1': [], 'test_f1': []}
    for n_learners in param_pairs:
        model = models.boosted_LR(estimator, n_learners=n_learners)
        results, _ = kfold_train_loop(data, model)
        storage['train_acc'].append(results['train']['accuracy'])
        storage['test_acc'].append(results['test']['accuracy'])
        storage['train_f1'].append(results['train']['f1'])
        storage['test_f1'].append(results['test']['f1'])

    experiment_name = ('{}-{}'.format(experiment_name, ('boosted' if boosted else 'not-boosted')))

    utils.plot_results(
        storage['train_acc'], 
        storage['test_acc'], 
        param_pairs,
        exp_name='{}_accuracy'.format(experiment_name), 
        yaxis='Accuracy'
    )

    utils.plot_results(
        storage['train_f1'], 
        storage['test_f1'], 
        param_pairs,
        exp_name='{}_f1'.format(experiment_name), 
        yaxis='F1'
    )

    return estimator, storage




def svm(data, n_estimators, experiment_name, estimator=None, boosted=False):
    """linear svm with and without adaboost"""
    # get the non-boosted model results
    if not estimator:
        model = models.SVM()
        _, single_best_model = kfold_train_loop(data, model)
        estimator = single_best_model.best_estimator_.named_steps['clf']

    # use optimal parameter C to generate param_pairs
    C = estimator.C
    param_pairs = [(C/n, (n if boosted else 1)) for n in n_estimators]

    storage = {'train_acc': [], 'test_acc': [], 'train_f1': [], 'test_f1': []}
    for C, n_learners in param_pairs:
        model = models.boosted_SVM(estimator, C=C, n_learners=n_learners)
        results, _ = kfold_train_loop(data, model)
        storage['train_acc'].append(results['train']['accuracy'])
        storage['test_acc'].append(results['test']['accuracy'])
        storage['train_f1'].append(results['train']['f1'])
        storage['test_f1'].append(results['test']['f1'])

    experiment_name = ('{}-{}'.format(experiment_name, ('boosted' if boosted else 'not-boosted')))

    utils.plot_results(
        storage['train_acc'],
        storage['test_acc'],
        param_pairs,
        exp_name='{}_accuracy'.format(experiment_name),
        yaxis='Accuracy'
    )

    utils.plot_results(
        storage['train_f1'],
        storage['test_f1'],
        param_pairs,
        exp_name='{}_f1'.format(experiment_name),
        yaxis='F1'
    )

    return estimator, storage


def decision_tree(data, n_estimators, experiment_name, estimator=None, boosted=False):
    """decision tree with and without adaboost"""
    if not estimator:
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
    max_depths = list(range(init_max_depth, init_max_depth-len(n_estimators)+1, -1))
    param_pairs = list(
        (zip(max_depths, n_estimators) if boosted else zip(max_depths, [1]*len(n_estimators)))
    )
    LOGGER.info('parameter pairs:\n{}'.format(param_pairs))

    storage = {'train_acc': [], 'test_acc': [], 'train_f1': [], 'test_f1': []}
    for max_depth, n_learners in param_pairs:
        model = models.random_forest(estimator, max_depth=max_depth, n_learners=n_learners)
        results, _ = kfold_train_loop(data, model)
        storage['train_acc'].append(results['train']['accuracy'])
        storage['test_acc'].append(results['test']['accuracy'])
        storage['train_f1'].append(results['train']['f1'])
        storage['test_f1'].append(results['test']['f1'])

    experiment_name = ('{}-{}'.format(experiment_name, ('boosted' if boosted else 'not-boosted')))

    utils.plot_results(
        storage['train_acc'],
        storage['test_acc'], param_pairs,
        exp_name='{}_accuracy'.format(experiment_name),
        yaxis='Accuracy'
    )

    utils.plot_results(
        storage['train_f1'],
        storage['test_f1'],
        param_pairs,
        exp_name='{}_f1'.format(experiment_name),
        yaxis='F1')

    return estimator, storage


def mlp(data, n_estimators, experiment_name, estimator=None, boosted=False):
    """mlp with and without adaboost"""
    # get the non-boosted model results
    if not estimator:
        model = models.mlp()
        _, single_model = kfold_train_loop(data, model)
        estimator = single_model.best_estimator_.named_steps['clf']

    # use optimal parameter C to generate param_pairs
    hidden_size = estimator.hidden_layer_sizes
    hidden_sizes = [int(np.floor(hidden_size / n)) for n in n_estimators]
    param_pairs = list(
        zip(hidden_sizes, (n_estimators if boosted else [1] * len(n_estimators)))
    )

    storage = {'train_acc': [], 'test_acc': [], 'train_f1': [], 'test_f1': []}
    for n_hid, n_learners in param_pairs:
        model = models.boosted_mlp(estimator, n_hid=n_hid, n_learners=n_learners)
        boosted_results, boosted_best_model = kfold_train_loop(data, model)
        storage['train_acc'].append(boosted_results['train']['accuracy'])
        storage['test_acc'].append(boosted_results['test']['accuracy'])
        storage['train_f1'].append(boosted_results['train']['f1'])
        storage['test_f1'].append(boosted_results['test']['f1'])

    experiment_name = ('{}-{}'.format(experiment_name, ('boosted' if boosted else 'not-boosted')))

    utils.plot_results(
        storage['train_acc'],
        storage['test_acc'],
        param_pairs,
        exp_name='{}_accuracy'.format(experiment_name),
        yaxis='Accuracy'
    )

    utils.plot_results(
        storage['train_f1'],
        storage['test_f1'],
        param_pairs,
        exp_name='{}_f1'.format(experiment_name),
        yaxis='F1'
    )

    return(estimator, storage)


