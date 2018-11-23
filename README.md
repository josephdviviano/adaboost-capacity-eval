ift6390-final
-------------

If you want to train a subset of the models, feel free to make a copy of `train_wine.py`
and build from that. They are all set up to work comme ca:`

+ `train\_????.py -v` : prints any messages in the code wrapped by `LOGGER.debug()` otherwise, you will only see messages from `LOGGER.info()`. Note all output will end up in `logs/`.
+ `train\_????.py -t` : the data import functions will use 450 training examples, and 50 validation examples, for quick training.
+ `train\_????.py -h` : print help.


**general structure**

+ `train.py`: runs a set of experiments from `experiments.py`, and write the results (e.g., performance, confusion matrix) to some files.
+ `models.py`: a collection of scikit learn models
+ `experiments.py`: the application of some models from `models.py` to the `data/`, to produce predictions.
+ `utils.py`: misc functions that might be useful in any of the other modules.

