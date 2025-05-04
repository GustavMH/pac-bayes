#!/usr/bin/env python3
import re
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import cache
from keras.datasets import imdb

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import sys
sys.path.insert(1, "RefineThenCalibrateVision")
from main import load_dataset
from majority_vote_bounds import optimize_rho, optimize_rho_oob

def group_strs(strs, regex=None):
    """Groups identity strs by the model number in the string"""
    assert(regex)

    ss = [(s, *re.search(regex, s).group(1,2)) for s in strs]
    ss = sorted(ss, key=lambda x: int(x[1]))

    res = []
    acc = None
    current_n = None
    for s, model_n, epoch_n in ss:
        if model_n == current_n:
            acc.append(s)
        else:
            current_n = model_n
            res.append(acc)
            acc = []
            acc.append(s)
    res.append(acc)
    res = res[1:]

    return np.array(res)


@cache
def pickle_cache(path):
    f = open(path, "rb")
    return pickle.load(f)


def best_pred_idx(preds, truth):
    """
    Given a sequence of predictions, evaluate for
    accuracy against y_true, return the index of best one
    """
    return np.nansum(preds.squeeze() == truth, axis=1).argmax()


def select_model_predictions(folder, mode="all", y_val=None, run_idx=0, regex=None):
    """Select a subset of model checkpoints from a folder"""
    candidate_strs = Path(folder).glob("*_val_predictions.pkl")
    candidate_strs = [p.name for p in candidate_strs]
    candidate_strs = sorted(candidate_strs)

    def get_model_files(val_name):
        base = re.match("(.+?)_val_predictions.pkl", val_name)[1]
        val = folder / val_name
        test = folder / f"{base}_test_predictions.pkl"
        return pickle_cache(val), pickle_cache(test)

    match mode:
        case "all":
            res = [get_model_files(s) for s in candidate_strs]
            val = [x[0] for x in res]
            test = [x[1] for x in res]
            return np.array(val), np.array(test)

        case "best checkpoints":
            candidate_strs = group_strs(candidate_strs, regex)
            preds_val  = np.array([[get_model_files(s)[0] for s in ss] for ss in candidate_strs])
            preds_test = np.array([[get_model_files(s)[1] for s in ss] for ss in candidate_strs])
            best_preds_idx = np.array([best_pred_idx(val, y_val) for val in preds_val])
            return (
                np.array([preds[i] for i, preds in zip(best_preds_idx, preds_val)]),
                np.array([preds[i] for i, preds in zip(best_preds_idx, preds_test)])
            )

        case "last checkpoints":
            candidate_strs = group_strs(candidate_strs, regex)
            last_strs = [models[-1] for models in candidate_strs]
            res = [get_model_files(s) for s in last_strs]
            val = [x[0] for x in res]
            test = [x[1] for x in res]
            return np.array(val), np.array(test)

        case "single run":
            candidate_strs = group_strs(candidate_strs, regex)[run_idx]
            res = [get_model_files(s) for s in candidate_strs]
            val = [x[0] for x in res]
            test = [x[1] for x in res]
            return np.array(val), np.array(test)


def one_hot(idxs, n_cats):
    res = np.zeros((idxs.size, n_cats))
    res[np.arange(idxs.size), idxs] = 1
    return res


def ensemble_predictions_MV(rho, predictions, n_cats):
    # predictions.shape = (model, prediction)
    pred = np.array([one_hot(p, n_cats) for p in predictions])
    # pred.shape = (model, prediction, n_cats)
    pred = (pred * rho[:, None, None]).sum(axis=0)
    # pred.shape = (prediction, n_cats)
    pred = np.argmax(pred, axis=1)
    # pred.shape = (prediction)
    return pred

def ensemble_predictions_AVG(rho, predictions, n_cats):
    # pred.shape = (model, prediction, n_cats)
    pred = (predictions * rho[:, None, None]).sum(axis=0)
    # pred.shape = (prediction, n_cats)
    pred = np.argmax(pred, axis=1)
    # pred.shape = (prediction)
    return pred


def select_model_preds_imdb(folder, configs, mode="all", run_idx=0):
    subfolders = sorted([conf.parents[0] for conf in configs])
    candidate_strs = []
    for subfolder in subfolders:
        val_preds = sorted([(subfolder, p.stem) for p in subfolder.glob("*.h5")])
        candidate_strs.append(val_preds)

    def get_model_files(path_stem_tuple):
        folder, base = path_stem_tuple
        val_indices = list(folder.glob("*config.json"))[0]
        val_indices = np.array(json.load(open(val_indices, "rb"))["val_indices"])
        val_sparse = folder / f"{base}_val_predictions.pkl"
        val_sparse = pickle_cache(val_sparse).squeeze()
        val = np.full(25000, np.nan)
        val[val_indices] = val_sparse
        test = folder / f"{base}_test_predictions.pkl"
        return val, pickle_cache(test)
    candidate_strs = np.array(candidate_strs)

    match mode:
        case "all":
            res = [get_model_files(s) for s in candidate_strs.reshape(-1,2)]
            val = [x[0] for x in res]
            test = [x[1] for x in res]
            return np.array(val), np.array(test)

        case "best checkpoints":
            preds_val  = np.array([[get_model_files(s)[0] for s in ss] for ss in candidate_strs])
            preds_test = np.array([[get_model_files(s)[1] for s in ss] for ss in candidate_strs])
            best_preds_idx = np.array([best_pred_idx(val, y_train) for val in preds_val])
            return (
                np.array([preds[i] for i, preds in zip(best_preds_idx, preds_val)]),
                np.array([preds[i] for i, preds in zip(best_preds_idx, preds_test)])
            )

        case "last checkpoints":
            last_strs = [models[-1] for models in candidate_strs]
            res = [get_model_files(s) for s in last_strs]
            val = [x[0] for x in res]
            test = [x[1] for x in res]
            return np.array(val), np.array(test)

        case "single run":
            res = [get_model_files(s) for s in candidate_strs[run_idx]]
            val = [x[0] for x in res]
            test = [x[1] for x in res]
            return np.array(val), np.array(test)

        case _:
            Exception(f"No mode named {mode}")

try:
    X_train
except Exception:
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=20000)

X_val = np.full(X_train.size, np.nan, dtype=object)
y_val = np.full(y_train.size, np.nan, dtype=object)
n_cats = 2

# for i in range(30):
folder = Path("CodeSubmission") / "imdb" / "results" / "imdb" / "bootstr"
configs = list(folder.glob("**/*config.json"))

conf_file = configs[0]
conf = json.load(open(conf_file, "rb"))
val_idxs = np.array(conf["val_indices"])
X_val[val_idxs] = X_train[val_idxs]
y_val[val_idxs] = y_train[val_idxs]

assert(5000 == sum([0 if isinstance(x, float) else 1 for x in X_val]))
assert(5000 == np.nansum(y_val >= 0))

select = "best checkpoints"

preds_val, preds_test = select_model_preds_imdb(folder, configs, mode=select)
preds_models_val = preds_val > 0.5
preds_models_test = preds_test > 0.5

def make_rho_ensemble(subset_idx):
    # TODO check oob tandem risks
    bound, rho, lam = optimize_rho_oob(preds_models_val[subset_idx], y_train, 5000)
    preds = np.array([one_hot(p, 2) for p in preds_models_test[subset_idx].squeeze().astype(np.int64)])
    preds_ensemble_test = ensemble_predictions_AVG(rho, preds, n_cats)
    return (preds_ensemble_test == y_test).mean()

def make_uniform_ensemble(subset_idx):
    rho = np.ones(subset_idx.size) / subset_idx.size
    preds = np.array([one_hot(p, 2) for p in preds_models_test[subset_idx].squeeze().astype(np.int64)])
    preds_ensemble_test = ensemble_predictions_AVG(rho, preds, n_cats)
    return (preds_ensemble_test == y_test).mean()

def make_early_stopping(subset_idx):
    # TODO fix
    val_scores = (preds_models_val == y_test).mean(1)
    preds_test = preds_models_test[val_scores.argmax()]
    return (preds_test == y_test).mean()

subsets = [np.random.choice(np.arange(50), i) for i in range(2,31)]
#subsets = [np.arange(0,i) for i in range(1,16)]
X = np.array([s.size for s in subsets])
rho_acc     = np.zeros(X.size)
uniform_acc = np.zeros(X.size)
early_acc   = np.zeros(X.size)

# Select different epoch spacing
early_acc   = [make_early_stopping(subset) for subset in subsets]
uniform_acc = [make_uniform_ensemble(subset) for subset in subsets]

select = "all"

preds_val, preds_test = select_model_preds_imdb(folder, configs, mode=select)
preds_models_val = preds_val > 0.5
preds_models_test = preds_test > 0.5
rho_acc     = [make_rho_ensemble(subset) for subset in subsets]

plt.title(f"Ensembles of '{select}', on IMDB")
plt.plot(X, rho_acc, label="PAC-Bayes ensemble")
plt.plot(X, uniform_acc, label="Uniform ensemble")
plt.plot(X, early_acc, label="Best model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("No. of models")
#plt.xscale("log")
plt.ylim(.65,.88)
plt.legend()
plt.savefig("ensemble.png")
plt.close()
