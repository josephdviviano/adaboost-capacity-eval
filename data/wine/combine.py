#!/usr/bin/python3
"""
Combine the red-wine and the white wine dataset into `data/wine/data.csv`

To run this script:
    python data/wine/combine.py
"""
import pandas

OUT_PATH = './data/wine/data.csv'
WHITE_WINE_PATH = './data/wine/winequality-white.csv'
RED_WINE_PATH = './data/wine/winequality-red.csv'

def load_data(path):
    """
    Load data into a pandas dataframe
    """
    return pandas.read_table(path, delimiter=';')

def save_data(dataframe):
    """
    Save data
    """
    dataframe.to_csv(OUT_PATH, index=False)

if __name__ == '__main__':
    white_wine = load_data(WHITE_WINE_PATH)
    red_wine = load_data(RED_WINE_PATH)
    dataframe = pandas.concat([white_wine, red_wine])
    dataframe.to_csv(OUT_PATH, index=False)
