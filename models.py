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
from sklearn.ensemble import AdaBoostClassifier
import argparse
import logging
import numpy as np
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

LOGGER = logging.getLogger(os.path.basename(__file__))

# global settings for all cross-validation runs
SETTINGS = {
    'n_cv': 50,
    'n_inner': 3,
}

# controls how chatty RandomizedCV is
VERB_LEVEL = 0

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

    return model
  
def SVM_nonlinear(data):
    """soft SVM with kernel"""
    LOGGER.debug('building SVM model')
    # hyperparameters to search for randomized cross validation
    settings = {
        'dim__n_components': stats.randint(1, 11),
        'clf__tol': stats.uniform(10e-5, 10e-1),
        'clf__C': stats.uniform(10e-3, 1)
    }

    # model we will train in our pipeline
    clf = SVC(gamma=0.001, max_iter=-1)

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return model

def boosted_SVM(data):
    """ baseline: linear classifier (without kernel)"""
    LOGGER.debug('building SVM model')
    # hyperparameters to search for randomized cross validation
    settings = {
        'dim__n_components': stats.randint(1, 9),
        'clf__n_estimators':[25,30,35,40,45,50],
        'clf__learning_rate':[0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56]         
    }

    # model we will train in our pipeline
    #clf = SVC(kernel='linear', max_iter=-1  )
    #clf = AdaBoostClassifier(SVC(C=stats.uniform(10e-3, 1), tol=stats.uniform(10e-5, 10e-1),probability=True, kernel='linear' ))
    clf = AdaBoostClassifier(SVC(C=0.01, tol=0.001,probability=True, kernel='linear' )) 
    
    
    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])
    
    # this will learn our best parameters for the final
    model = RandomizedSearchCV(pipe, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return model

def _tree():
    settings = {
        'dim__n_components': stats.randint(4, 8),
        'clf__max_depth': stats.randint(1, 12),
        'clf__min_samples_split': stats.randint(2, 5),
        'clf__min_samples_leaf': stats.randint(1, 5),
        'clf__max_features': stats.randint(1, 5)
    }

    clf = DecisionTreeClassifier(criterion='entropy')

    pipeline = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])
    
    model = RandomizedSearchCV(pipeline, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy')

    return model  

def _forest(adaboost=True):
    
    settings = {
        'dim__n_components': stats.randint(4, 8),
        'clf__n_estimators': stats.randint(25, 50),
        'clf__learning_rate': stats.uniform(0.1, 2.)
    }

    clf = AdaBoostClassifier(DecisionTreeClassifier(
        criterion='entropy', max_depth=8, max_features='auto'))

    pipeline = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])
    
    model = RandomizedSearchCV(pipeline, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return model

def decision_tree(adaboost=True):
    """
    Build decision tree model
    """
    LOGGER.info('Building decision tree model with adaboost set to: {}'.format(adaboost))
    if adaboost:
        model = _forest()
    else:
        model = _tree()
    return model
