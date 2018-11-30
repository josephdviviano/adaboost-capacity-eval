#!/usr/bin/python3
"""
Split data into a train and a test set. i.e. the script creates

    ./data/<DATASET>/ train.csv test.csv

where <DATASET> is either wine or ...

To run this script:
    python data/split.py --data red-wine --train_size 0.85
"""
import argparse
import random
import pickle
import pandas

def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description='Train test split')
    parser.add_argument('--train_size', type=float, default=0.85)
    parser.add_argument('--data', type=str, choices=['wine'])
    return parser.parse_args()

def load_data(path):
    """
    Load data into a pandas dataframe
    """
    return pandas.read_table(path, delimiter=',')

def _split_indices(data, train_size):
    storage = {'train': [], 'test': []}
    data_size = len(data)
    train_size = round(data_size * train_size)
    examples = range(len(data))
    storage['train'] = random.sample(examples, train_size)
    storage['test'] = [ex for ex in examples if ex not in storage['train']]
    return storage

def split_data(dataframe, train_size):
    """
    Split the data into a training set
    and a test set according to 'train_size'

    Args:
        dataframe: (pandas.Dataframe)
        split: (list of float) train/valid/test split 
    """
    split_idx = _split_indices(dataframe, train_size)
    train_data = dataframe.iloc[split_idx['train']]
    test_data = dataframe.iloc[split_idx['test']]
    return train_data, test_data

def save_data(dataframe, args, train=True):
    """
    Save a dictionary data to a pickle file
    """
    data = 'train' if train else 'test'
    out_dir = './data/{}/{}.csv'.format(args.data, data)
    dataframe.to_csv(out_dir, index=False)

def main(args):
    """
    wrapper - create the data<spam_percentage>.csv file
    """
    data_path = './data/{}/data.csv'.format(args.data)
    data = load_data(data_path)
    train_data, test_data = split_data(data, args.train_size)
    
    save_data(train_data, args, train=True)
    save_data(test_data, args, train=False)

if __name__ == '__main__':
    main(argparser())
