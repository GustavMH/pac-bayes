#!/usr/bin/env python3

try:
    ds
except NameError:
    ds = dict(np.load("/home/gustav/Downloads/pac-bayes-predictions/imdb_predictions.npz"))
