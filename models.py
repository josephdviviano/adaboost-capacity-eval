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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

LOGGER = logging.getLogger(os.path.basename(__file__))

# global settings for all cross-validation runs
SETTINGS = {
    'n_cv': 25,
    'n_inner': 3,
}

# controls how chatty RandomizedCV is
VERB_LEVEL = 0

def decision_tree(data, adaboost=True):
    LOGGER.debug('building decision tree model')
    # hyperparameters to search for randomized cross validation
    settings = {
        'dim__n_components': stats.randint(1, 11),
        #'clf__max_depth': stats.randint(1, 12),
        #'clf__min_samples_split': stats.randint(2, 5),
        #'clf__min_samples_leaf': stats.randint(1, 5),
        #'clf__max_features': stats.randint(1, 12)
        #'clf__n_estimators':[25,30,35,40,45,50],
        #'clf__learning_rate':[0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56]
        
    }

    clf = AdaBoostClassifier(DecisionTreeClassifier())

    pipeline = Pipeline([
        ('pre', StandardScaler()),
        ('dim', PCA(svd_solver='randomized')),
        ('clf', clf),
    ])

    # this will learn our best parameters for the final model
    model = RandomizedSearchCV(pipeline, settings, n_jobs=-1, verbose=VERB_LEVEL,
        n_iter=SETTINGS['n_cv'], cv=SETTINGS['n_inner'], scoring='accuracy'
    )

    return model


