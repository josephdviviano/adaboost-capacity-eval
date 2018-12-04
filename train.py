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

    if args.model not in ['tree', 'mlp', 'svm', 'all']:
        LOGGER.error('invalid experiment submitted -m {decision_tree, svm, mlp, all}')
        sys.exit(1)

    # load our the datasets
    if args.data == 'wine':
        data = utils.load_wine(args.test)
    if args.data == 'covtype':
        data = utils.load_covertype(args.test)
    if args.data == 'covtype_balanced':
        data = utils.load_covertype(args.test, balanced=True)

    if args.model == 'tree' or args.model == 'all':
        experiment_name = '{}-{}'.format('tree', args.data)
        n_estimators = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        estimator = exp.decision_tree(data, n_estimators, experiment_name, boosted=False)
        exp.decision_tree(data, n_estimators, experiment_name, estimator, boosted=True)

    if args.model == 'svm' or args.model == 'all':
        experiment_name = '{}-{}'.format('svm', args.data)
        n_estimators = [1, 2, 4, 5, 10, 20]
        svm_pred, svm_model = exp.svm(data, n_estimators, experiment_name, boosted=False)
        svm_pred, svm_model = exp.svm(data, n_estimators, experiment_name, boosted=True)

    if args.model == 'mlp' or args.model == 'all':
        experiment_name = '{}-{}'.format('mlp', args.data)
        n_estimators = [1, 2, 4, 5, 10, 20]
        results = exp.mlp(data, n_estimators, experiment_name, boosted=False)
        results = exp.mlp(data, n_estimators, experiment_name, boosted=True)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-v", "--verbose", action="store_true", 
                            help="increase output verbosity")
    argparser.add_argument("-t", "--test", action="store_true", 
                            help="training set size=500")
    argparser.add_argument("-m", "--model", choices=['tree', 'svm', 'mlp'],
                            help="which model?")
    argparser.add_argument("-d", "--data", choices=['wine', 'covtype', 'covtype_balanced'], 
                            help="which data?")
    args = argparser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)

    main(args)


