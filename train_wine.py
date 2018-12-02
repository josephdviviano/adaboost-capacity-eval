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

# adds a simple logger
logging.basicConfig(level=logging.INFO, format="[%(name)s:%(funcName)s:%(lineno)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger()

category_map = {}

def main(args):

    log_fname = "logs/{}.log".format(__name__)
    if os.path.isfile(log_fname):
        os.remove(log_fname)
    log_hdl = logging.FileHandler(log_fname)
    log_hdl.setFormatter(logging.Formatter('%(message)s'))
    LOGGER.addHandler(log_hdl)

    data = utils.load_wine(args.test)

    if args.model == 'decision_tree':
<<<<<<< HEAD
        param_pairs = [(6, 1), (5, 2), (4, 4), (3, 8), (2, 16)]#, (5, 32), (4, 64), (3, 128)]
        results = exp.decision_tree(data, param_pairs)
=======
        param_pairs = [(10, 1), (8, 20), (6, 30), (4, 40), (2, 50)]
        results, best_model = exp.decision_tree(data, param_pairs)

>>>>>>> f47478eed5a9ee09a67eb1f8b2b1d6f56ba87c96
    elif args.model == 'svm':
        svm_pred, svm_model = exp.svm(data)

    elif args.model == 'nn':
        single_pred, boosted_pred, single_model, boosted_model = exp.nn(data)

    else:
        LOGGER.warning('invalid experiment submitted -m {decision_tree, svm, nn}')
        sys.exit(1)


    utils.write_results('results/wine.csv', results)


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


