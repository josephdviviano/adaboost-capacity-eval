#!/bin/bash

#./train.py -v -m tree -d wine
#./train.py -v -m tree -d covtype
#./train.py -v -m tree -d covtype_balanced
./train.py -v -m svm -d wine
./train.py -v -m svm -d covtype
#./train.py -v -m svm -d covtype_balanced
./train.py -v -m mlp -d wine
./train.py -v -m mlp -d covtype
#./train.py -v -m mlp -d covtype_balanced

