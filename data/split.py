#!/usr/bin/python3
"""
Split data into a train and a test set. i.e. the script creates

    ./data/<DATASET>/ train.csv test.csv

where <DATASET> is either red-wine or ...

To run this script:
    python data/split.py --data red-wine --train_size 0.85
"""
import argparse
import random
import pickle
import pandas

ORIGINAL_DATA_PATH = './original/smsSpamCollection.csv'

def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description='Image classifier implementation')
    parser.add_argument('--spam_percentage', '-spam', type=float, default=None)
    parser.add_argument('--train_size', type=float, default=0.75)
    parser.add_argument('--valid_size', type=float, default=0.15)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--out_dir', type=str, default='.')
    return parser.parse_args()

def load_data(path):
    """
    Load data into a pandas dataframe
    """
    return pandas.read_table(path, delimiter=',')

def select_spam(dataframe, spam_percentage):
    """
    Keep `spam_percentage` of the spam and
    remove the rest. Return the new dataframe
    """
    if spam_percentage:
        spam_idx = dataframe.index[dataframe['target'] == 'spam'].tolist()
        len_ham = len(dataframe) - len(spam_idx)
        len_spam = int(spam_percentage * len_ham / (1 - spam_percentage))
        idx_to_remove = random.sample(spam_idx, len(spam_idx) - len_spam)
        dataframe = dataframe.drop(idx_to_remove, axis=0)
    return dataframe

def _split_indices(data, split):
    storage = {'train': [], 'valid': [], 'test': []}
    data_size = len(data)
    train_size = round(data_size * split[0])
    valid_size = round(data_size * split[1])
    examples = range(len(data))
    storage['train'] = random.sample(examples, train_size)
    examples = [ex for ex in examples if ex not in storage['train']]
    storage['valid'] = random.sample(examples, valid_size)
    storage['test'] = [ex for ex in examples if ex not in storage['valid']]
    return storage

def split_data(dataframe, split):
    """
    Split the data into a training set
    and a test set according to 'train_size'

    Args:
        dataframe: (pandas.Dataframe)
        split: (list of float) train/valid/test split 
    """
    split_idx = _split_indices(dataframe, split)
    train_data = dataframe.iloc[split_idx['train']]
    valid_data = dataframe.iloc[split_idx['valid']]
    test_data = dataframe.iloc[split_idx['test']]
    return train_data, valid_data, test_data

def save_data(dataframe, name, args):
    """
    Save a dictionary data to a pickle file
    """
    out_dir = '{}/{}{}.csv'.format(
        args.out_dir, name, args.spam_percentage)
    dataframe.to_csv(out_dir, index=False)

def main(args):
    """
    wrapper - create the data<spam_percentage>.csv file
    """
    split = [args.train_size, args.valid_size, args.test_size]
    data = load_data(ORIGINAL_DATA_PATH)
    data = select_spam(data, args.spam_percentage)
    train_data, valid_data, test_data = split_data(data, split)
    
    if not args.spam_percentage:
        args.spam_percentage = ''
    
    save_data(train_data, 'train', args)
    save_data(valid_data, 'valid', args)
    save_data(test_data, 'test', args)

if __name__ == '__main__':
    main(argparser())
