ift6390-final
-------------

## Decision tree experiment
```
python train --model decision_tree --adaboost True
```
## SVM experiment
```
python train_wine --model svm --adaboost True
```

If you want to train a subset of the models, feel free to make a copy of `train_wine.py`
and build from that. They are all set up to work comme ca:`

+ `train.py -v` : prints any messages in the code wrapped by `LOGGER.debug()` otherwise, you will only see messages from `LOGGER.info()`. Note all output will end up in `logs/`.
+ `train.py -t` : the data import functions will use 450 training examples, and 50 validation examples, for quick training.
+ `train.py -m ???` : runs model ??? {svm, decision_tree, mlp}
+ `train.py -h` : print help.

**general structure**

+ `train.py`: runs a set of experiments from `experiments.py`, and write the results (e.g., performance, confusion matrix) to some files.
+ `models.py`: a collection of scikit learn models
+ `experiments.py`: the application of some models from `models.py` to the `data/`, to produce predictions.
+ `utils.py`: misc functions that might be useful in any of the other modules.
+ `data/wine/data.py`: contains the concatenation of _white wine_ and the _red wine_ datasets
+ `data.wine.combine`: script that combines the _white_ and _red_ wine data
+ `data.split.py`: split a specify data (CSV) in a train and a test file. (see the docstring for more details)
+ `neural_network/`: custom modification of the neural network class from scikit learn to accept class_weights.
**NOTE**: The data is already combined and split so there is no need to run `combine.py` nor `split.py`

