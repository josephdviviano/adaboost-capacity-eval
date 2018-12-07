"""
holds our models (e.g., imagenet, cnns, etc, to be imported into experiments.py)
"""
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning) # adaboost
#warnings.filterwarnings("ignore", category=ConvergenceWarning) # svc
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.linear_model import LogisticRegression
from neural_network import MLPClassifier # custom mlp accepts sample weights
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import argparse
import logging
import numpy as np
import os


LOGGER = logging.getLogger(os.path.basename(__file__))

# use logger to figure out verbosity for RandomizedCV
#if LOGGER.getEffectiveLevel() != 20:
#    VERB_LEVEL = 2
#else:
VERB_LEVEL = 0

# global settings for all cross-validation runs
SETTINGS = {
    'n_cv': 100,
    'n_folds': 10,
    'ada_lr': stats.uniform(10e-5, 10e-1),
    'cv_score': 'f1_macro'
}


def SVM():
    """ baseline: linear classifier (without kernel)"""
    LOGGER.debug('building SVM model')

    # hyperparameters to search for randomized cross validation
    settings = {
        'clf__tol': stats.uniform(10e-3, 10e-1),
        'clf__C': stats.uniform(10e-3, 1)
    }

    # model we will train in our pipeline
    clf = SVC(kernel='linear', max_iter=2e3)

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_folds'], scoring=SETTINGS['cv_score']
    )

    return(model)


def boosted_SVM(estimator, C=1, n_learners=10):
    """ baseline: linear classifier (without kernel)"""
    settings = {
        'clf__learning_rate': SETTINGS['ada_lr']
    }

    # set up this estimator's C parameter and adaboost
    LOGGER.debug('setting up adaboost with C={}, n_estimators={}'.format(
        C, n_learners))
    estimator.C = C
    clf = AdaBoostClassifier(
        base_estimator=estimator, n_estimators=n_learners, algorithm='SAMME'
    )

    pipeline = Pipeline([
        ('pre', StandardScaler()),
        ('clf', clf),
    ])

    model = RandomizedSearchCV(pipeline, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_folds'], scoring=SETTINGS['cv_score']
    )

    return(model)


def decision_tree(data):
    """
    decision tree model
    """
    LOGGER.debug('building decision tree model')
    n_features = data['X']['train'].shape[1]
    n_samples = data['X']['train'].shape[0]
    min_samples_split = [2, int(0.01*n_samples)]
    min_samples_leaf = [1, int(0.005 * n_samples)]

    LOGGER.info('Setting for the decision tree model:')
    LOGGER.info('min_samples_split: {}'.format(min_samples_split))
    LOGGER.info('min_samples_leaf: {}'.format(min_samples_leaf))

    settings = {
        'clf__max_depth': stats.randint(1, n_features),
        'clf__min_samples_split': stats.randint(*min_samples_split),
        'clf__min_samples_leaf': stats.randint(*min_samples_leaf),
        'clf__max_features': stats.randint(1, n_features),
        'clf__min_impurity_decrease': stats.uniform(0., 0.1)
    }

    clf = DecisionTreeClassifier(criterion='entropy')

    pipeline = Pipeline([
        ('pre', StandardScaler()),
        ('clf', clf),
    ])

    model = RandomizedSearchCV(pipeline, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_folds'], scoring=SETTINGS['cv_score']
    )

    return model


def random_forest(base_model, max_depth, n_learners):
    settings = {
        'booster__learning_rate': SETTINGS['ada_lr']
    }

    # set up max_depth and n_learners (per-model parameters vs n in ensemble)
    LOGGER.debug('setting up adaboost with max_depth={}, n_estimators={}'.format(
        max_depth, n_learners))
    base_model.max_depth = max_depth
    booster = AdaBoostClassifier(base_estimator=base_model, n_estimators=n_learners)

    pipeline = Pipeline([
        ('pre', StandardScaler()),
        ('booster', booster),
    ])

    model = RandomizedSearchCV(pipeline, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_folds'], scoring=SETTINGS['cv_score']
    )

    return(model)


def mlp():
    """build a not-boosted MLP classifier"""
    LOGGER.debug('building mlp model')

    # alpha = l2 regularization, reciprocal == log-uniform distribution
    settings = {
        'clf__alpha': stats.reciprocal(10e-6, 10e-1),
        'clf__learning_rate_init': stats.reciprocal(10e-6, 10e-1),
        'clf__hidden_layer_sizes': stats.randint(40, 1000)
    }

    clf =  MLPClassifier(
        activation='relu', solver='sgd',  batch_size=32, early_stopping=True,
        learning_rate='invscaling', momentum=0.9, nesterovs_momentum=True
    )

    pipeline = Pipeline([
        ('pre', StandardScaler()),
        ('clf', clf),
    ])

    model = RandomizedSearchCV(pipeline, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_folds'], scoring=SETTINGS['cv_score']
    )

    return(model)


def boosted_mlp(estimator, n_hid=10, n_learners=10):
    """uses model learned by mlp_single for the model settings"""
    settings = {
        'clf__learning_rate': SETTINGS['ada_lr']
    }

    # set hidden layer size on estimator, and set up adaboost
    LOGGER.debug('setting up adaboost with n_hid={}, n_estimators={}'.format(
        n_hid, n_learners))
    estimator.hidden_layer_sizes = n_hid
    clf = AdaBoostClassifier(
        base_estimator=estimator, n_estimators=n_learners, algorithm='SAMME'
    )

    pipeline = Pipeline([
        ('pre', StandardScaler()),
        ('clf', clf),
    ])

    model = RandomizedSearchCV(pipeline, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_folds'], scoring=SETTINGS['cv_score']
    )

    return(model)


