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
    if args.data == 'wine':
        data = utils.load_wine(args.test)
    if args.data == 'covtype':
        data = utils.load_covertype(args.test)
    if args.data == 'covtype_balanced':
        data = utils.load_covertype(args.test, balanced=True)

    if args.model == 'tree' or args.model == 'all':
        experiment_name = '{}-{}'.format('tree', args.data)
        n_estimators = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        single_model, result = exp.decision_tree(data, n_estimators, experiment_name, boosted=False)
        _, result = exp.decision_tree(data, n_estimators, experiment_name, estimator=single_model, boosted=True)

    if args.model == 'lr' or args.model == 'all':
        experiment_name = '{}-{}'.format('lr', args.data)
        n_estimators = [1, 2, 4, 5, 10, 20]
        single_model, result = exp.logistic_regression(data, n_estimators, experiment_name, boosted=False)
        _, result = exp.logistic_regression(data, n_estimators, experiment_name, estimator=single_model, boosted=True)

    if args.model == 'mlp' or args.model == 'all':
        experiment_name = '{}-{}'.format('mlp', args.data)
        n_estimators = [1, 2, 4, 5, 10, 20]
        single_model, result = exp.mlp(data, n_estimators, experiment_name, boosted=False)
        _, result = exp.mlp(data, n_estimators, experiment_name, estimator=single_model, boosted=True)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-v", "--verbose", action="store_true",
                            help="increase output verbosity")
    argparser.add_argument("-t", "--test", action="store_true",
                            help="training set size=500")
    argparser.add_argument("-m", "--model", choices=['tree', 'lr' , 'mlp', 'all'],
                            help="which model?")
    argparser.add_argument("-d", "--data", choices=['wine', 'covtype', 'covtype_balanced'],
                            help="which data?")
    args = argparser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)

    main(args)


