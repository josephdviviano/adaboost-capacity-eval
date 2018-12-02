"""
holds our models (e.g., imagenet, cnns, etc, to be imported into experiments.py)
"""
from scipy import stats
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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
if LOGGER.getEffectiveLevel() != 20:
    VERB_LEVEL = 2
else:
    VERB_LEVEL = 0

# global settings for all cross-validation runs
SETTINGS = {
    'n_cv': 50,
    'n_folds': 10,
    'ada_lr': stats.uniform(10e-5, 10e-1)
}


def SVM():
    """ baseline: linear classifier (without kernel)"""
    LOGGER.debug('building SVM model')

    # hyperparameters to search for randomized cross validation
    settings = {
        'clf__tol': stats.uniform(10e-5, 10e-1),
        'clf__C': stats.uniform(10e-3, 1)
    }

    # model we will train in our pipeline
    clf = SVC(kernel='linear', max_iter=100)

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_folds'], scoring='accuracy'
    )

    return(model)


def boosted_SVM():
    """ baseline: linear classifier (without kernel)"""
    LOGGER.debug('building boosted SVM model')

    # hyperparameters to search for randomized cross validation
    settings = {
        'clf__n_estimators': [25,30,35,40,45,50],
        'clf__learning_rate': SETTINGS['ada_lr']
    }

    # model we will train in our pipeline
    #clf = SVC(kernel='linear', max_iter=-1  )
    #clf = AdaBoostClassifier(SVC(C=stats.uniform(10e-3, 1), tol=stats.uniform(10e-5, 10e-1),probability=True, kernel='linear' ))
    clf = AdaBoostClassifier(SVC(C=0.01, tol=0.001, probability=True, kernel='linear'))


    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_folds'], scoring='accuracy'
    )

    return(model)


def decision_tree():
    settings = {
        'clf__max_depth': stats.randint(2, 9),
        #'clf__min_samples_split': stats.randint(2, 5),
        #'clf__min_samples_leaf': stats.randint(1, 10),
        #'clf__max_features': stats.randint(2, )
    }

    clf = DecisionTreeClassifier(criterion='entropy')

    pipeline = Pipeline([
        ('pre', StandardScaler()),
        ('clf', clf),
    ])

    model = RandomizedSearchCV(pipeline, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=100, cv=SETTINGS['n_inner'], scoring='accuracy')


def random_forest(base_model, max_depth, n_learners):

    settings = {
        'clf__learning_rate': stats.uniform(0.5, 1.5)
    }

    base_model.max_depth = max_depth

    clf = AdaBoostClassifier(base_estimator=base_model, n_estimators=n_learners)

    pipeline = Pipeline([
        ('pre', StandardScaler()),
        ('clf', clf),
    ])

    model = RandomizedSearchCV(pipeline, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_folds'], scoring='accuracy'
    )

    return(model)


def mlp(n_hid=100):
    """build a not-boosted MLP classifier"""
    # estimate number of paramaters for single MLP assuming 10 features
    # nb: 10 boosters with hid = 10 has 220*10 parameters
    # vs: 1 model with hid = 100 has 10*100 + 100*10 + 100 + 10 = 2110 params

    # alpha = l2 regularization, reciprocal == log-uniform distribution
    settings = {
        'clf__alpha': stats.reciprocal(10e-6, 10e-1),
        'clf__learning_rate_init': stats.reciprocal(10e-6, 10e-1)
    }

    clf =  MLPClassifier(
        activation='relu', solver='sgd',  batch_size=32,
        hidden_layer_sizes=(100), learning_rate='invscaling', momentum=0.9,
        nesterovs_momentum=True, early_stopping=True
    )

    pipeline = Pipeline([
        ('pre', StandardScaler()),
        ('clf', clf),
    ])

    model = RandomizedSearchCV(pipeline, settings,
        n_jobs=-1, verbose=VERB_LEVEL, n_iter=SETTINGS['n_cv'],
        cv=SETTINGS['n_folds'], scoring='accuracy'
    )

    return(model)


def boosted_mlp(model):
    """uses model learned by mlp_single for the model settings"""
    settings = {
        'clf__learning_rate': SETTINGS['ada_lr']
    }

    # now set hidden layer size to 10
    estimator = model.best_estimator_.named_steps['clf']
    estimator.hidden_layer_sizes = 10
    clf = AdaBoostClassifier(
        base_estimator=estimator, n_estimators=10, algorithm='SAMME'
    )

    pipeline = Pipeline([
        ('pre', StandardScaler()),
        ('clf', clf),
    ])

    model = RandomizedSearchCV(pipeline, settings,
        n_jobs=-1, verbose=VERB_LEVEL, n_iter=SETTINGS['n_cv'],
        cv=SETTINGS['n_folds'], scoring='accuracy'
    )

    return(model)


