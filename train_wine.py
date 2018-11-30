#!/usr/bin/env python
"""
imports a set of experiments from experiments.py, runs them, and write results
"""

import matplotlib
matplotlib.use('agg')

import utils
import argparse
import experiments as exp
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

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

    # run experiments
    # TODO: add other models
    if args.model == 'decision_tree':
        results, best_model = exp.decision_tree(data, args.adaboost)
        #print(best_model.cv_results_)
        utils.write_results('results/wine.csv', results)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    argparser.add_argument("-t", "--test", action="store_true", help="training set size=500")
    argparser.add_argument("-m", "--model", help="which model?")
    argparser.add_argument("-ada", "--adaboost", help="Should adaboost be used?")
    args = argparser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)

    main(args)
