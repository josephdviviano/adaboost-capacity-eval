"""
holds our models (e.g., imagenet, cnns, etc, to be imported into experiments.py)
"""
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import argparse
import logging
import numpy as np
import os
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

LOGGER = logging.getLogger(os.path.basename(__file__))

# global settings for all cross-validation runs
SETTINGS = {
    'n_cv': 25,
    'n_inner': 3,
}

# controls how chatty RandomizedCV is
VERB_LEVEL = 0

def SVM_nonlinear(data):
    """soft SVM with kernel"""
    LOGGER.debug('building SVM model')
    # hyperparameters to search for randomized cross validation
    settings = {
        'dim__n_components': stats.randint(10, 1000),
        'clf__tol': stats.uniform(10e-5, 10e-1),
        'clf__C': stats.uniform(10e-3, 1)
    }

    # model we will train in our pipeline
    clf = SVC(gamma=0.001, max_iter=100)

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return(model)

def SVM(data):
    """ baseline: linear classifier (without kernel)"""
    LOGGER.debug('building SVM model')
    # hyperparameters to search for randomized cross validation
    settings = {
        'dim__n_components': stats.randint(10,1000),
        'clf__tol': stats.uniform(10e-5, 10e-1),
        'clf__C': stats.uniform(10e-3, 1)
    }

    # model we will train in our pipeline
    clf = SVC(kernel='linear', max_iter=100)

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return(model)


def Decision_tress(data, adaboost=True):
    LOGGER.debug('building decision tree model')
    # hyperparameters to search for randomized cross validation
    settings = {
        'clf__max_depth': stats.randint(1,12)
        #'clf__min_samples_split': stats.randint(1,5),
        #'clf__min_samples_leaf': stats.randint(1,5),
        #'clf__max_features': stats.randint(1,12)
    }
    #Not used: max_features, criterion, splitter, random_stat, max_leaf_nodes,min_samples_split, clf__min_samples_leaf, min_impurity_decrease, min_impurity_split, class_weight

    # model we will train in our pipeline
    clf = DecisionTreeRegressor()

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return(model)


