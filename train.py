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

    # load our two datasets
    wine_data = utils.load_wine(args.test)
    covt_data = utils.load_covertype(args.test)

    if args.model == 'decision_tree':
        #param_pairs = [(6, 1), (5, 5), (4, 10), (3, 15), (2, 20), (1, 25)]
        param_pairs = [(6, 1), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6)]
        results = exp.decision_tree(wine_data, param_pairs, 'tree-wine')
        results = exp.decision_tree(covt_data, param_pairs, 'tree-covt')

    elif args.model == 'svm':
        svm_pred, svm_model = exp.svm(wine_data, 'svm-wine')
        svm_pred, svm_model = exp.svm(wine_data, 'svm-covt')

    elif args.model == 'mlp':
        param_pairs = [(100, 1), (50, 2), (25, 4), (20, 5), (10, 10), (5, 20)]
        results = exp.mlp(wine_data, param_pairs, 'mlp-wine')
        results = exp.mlp(wine_data, param_pairs, 'mlp-covt')

    else:
        LOGGER.warning('invalid experiment submitted -m {decision_tree, svm, mlp}')
        sys.exit(1)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    argparser.add_argument("-t", "--test", action="store_true", help="training set size=500")
    argparser.add_argument("-m", "--model", help="which model?")
    argparser.add_argument("-ada", "--adaboost", type=bool, default=False, help="Should adaboost be used?")
    args = argparser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)

    main(args)


