#!/usr/bin/env python
"""
imports a set of experiments from experiments.py, runs them, and write results
"""
import matplotlib
matplotlib.use('agg')

import argparse
import experiments as exp
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import utils

logging.basicConfig(level=logging.INFO, format="[%(name)s:%(funcName)s:%(lineno)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger()

def main(args):

    log_fname = "logs/{}.log".format(__name__)
    if os.path.isfile(log_fname):
        os.remove(log_fname)
    log_hdl = logging.FileHandler(log_fname)
    log_hdl.setFormatter(logging.Formatter('%(message)s'))
    LOGGER.addHandler(log_hdl)

    # load our the datasets
    wine_data = utils.load_wine(args.test)
    #covt_data = utils.load_covertype(args.test)

    if args.model not in ['decision_tree', 'mlp', 'svm', 'all']:
        LOGGER.error('invalid experiment submitted -m {decision_tree, svm, mlp, all}')
        sys.exit(1)

    if args.model == 'decision_tree' or args.model == 'all':
        n_estimators = [1, 2, 4, 5, 10, 20]
        results = exp.decision_tree(wine_data, n_estimators, 'tree-wine', boosted=False)
        results = exp.decision_tree(wine_data, n_estimators, 'tree-wine', boosted=True)
        #results = exp.decision_tree(covt_data, n_estimators, 'tree-covt')

    if args.model == 'svm' or args.model == 'all':
        n_estimators = [1, 2, 4, 5, 10, 20]
        svm_pred, svm_model = exp.svm(wine_data, n_estimators, 'svm-wine')
        svm_pred, svm_model = exp.svm(covt_data, n_estimators, 'svm-covt')

    if args.model == 'mlp' or args.model == 'all':
        n_estimators = [1, 2, 4, 5, 10, 20]
        results = exp.mlp(wine_data, n_estimators, 'mlp-wine')
        results = exp.mlp(covt_data, n_estimators, 'mlp-covt')


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    argparser.add_argument("-t", "--test", action="store_true", help="training set size=500")
    argparser.add_argument("-m", "--model", help="which model?")
    argparser.add_argument("-d", "--data", choices=['wine', 'covtype'], help="which data?")
    args = argparser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)

    main(args)


