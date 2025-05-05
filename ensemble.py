#!/usr/bin/env python3
import re
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import cache
from majority_vote_bounds import optimize_rho, optimize_rho_oob

def best_pred_idx(preds, truth):
    return np.nansum(preds.squeeze() == truth, axis=1).argmax()

def ensemble_predictions_AVG(rho, predictions):
    pred = (predictions * rho[:, None, None]).sum(axis=0)
    pred = np.argmax(pred, axis=1)
    return pred

def nan_argmax(arr):
    # NOTE np.argmax will find a return value in [nan,nan]
    res = np.full(arr.shape[:-1], np.nan)
    idxs = ~np.isnan(arr).any(axis=-1)
    argmax = np.argmax(arr, axis=-1)
    res[idxs] = argmax[idxs]
    return res

def make_rho_ensemble(subset_idx, dataset):
    preds_test = dataset["predictions_test"][*subset_idx.T]
    preds_val = dataset["predictions_validation"][*subset_idx.T]
    preds_val = nan_argmax(preds_val)
    bound, rho, lam = optimize_rho_oob(preds_val, dataset["labels_validation"])
    preds_ensemble_test = ensemble_predictions_AVG(rho, preds_test)
    return (preds_ensemble_test == dataset["labels_test"]).mean()


def make_uniform_ensemble(subset_idx, dataset):
    preds_test = dataset["predictions_test"][*subset_idx.T]
    rho = np.ones(len(subset_idx)) / len(subset_idx)
    preds_ensemble_test = ensemble_predictions_AVG(rho, preds_test)
    return (preds_ensemble_test == dataset["labels_test"]).mean()


def chkpnt_acc(chkpnt, val_labels):
    idxs = ~np.isnan(chkpnt).any(axis=-1)
    return np.mean(chkpnt[idxs].argmax(-1) == val_labels[idxs])


def make_early_stopping(subset_idx, dataset):
    preds_val = dataset["predictions_validation"][*subset_idx.T]
    preds_test = dataset["predictions_test"][*subset_idx.T]
    labels_val = dataset["labels_validation"]
    labels_test = dataset["labels_test"]

    val_scores = np.array([chkpnt_acc(chkpnt, labels_val) for chkpnt in preds_val])
    preds_test = preds_test[val_scores.argmax()]
    return (preds_test.argmax(-1) == labels_test).mean()


def select_models(dataset, mode):
    n_models, n_checkpoints, _, n_cats = dataset["predictions_test"].shape
    match mode:
        case "all":
            return np.array([(i,j) for i in range(n_models) for j in range(n_checkpoints)])

        case "last":
            return np.array([(i,n_checkpoints-1) for i in range(n_models)])

        case "best":
            val_labels = dataset["labels_validation"]
            val_preds = dataset["predictions_validation"]

            model_idx = np.array([
                np.argmax([chkpnt_acc(chkpnt, val_labels)
                           for chkpnt in model])
                for model in val_preds
            ])

            return np.array((np.arange(len(model_idx)), model_idx)).T

imdb = np.load("imdb_predictions.npz")
idxs = np.array(select_models(imdb, "all"))
idxs_best = np.array(select_models(imdb, "best"))

n_iter = 5
X = np.arange(2,31)

rho_acc = np.zeros((n_iter, len(X)))
uni_acc = np.zeros((n_iter, len(X)))
best_acc = np.zeros((n_iter, len(X)))

for i in range(n_iter):
    print(i)
    subsets_models = [np.random.choice(np.arange(50),i)for i in X]
    subsets_chkpnt = [np.array([(i,j) for i in s for j in np.arange(10)]) for s in subsets_models]

    rho_acc[i] = np.array([make_rho_ensemble(subset, imdb) for subset in subsets_chkpnt])
    uni_acc[i] = np.array([make_uniform_ensemble(subset, imdb) for subset in subsets_chkpnt])
    best_acc[i] = np.array([make_rho_ensemble(idxs_best[subset], imdb) for subset in subsets_models])

plt.plot(X,uni_acc.mean(0), label="Uniform ensemble, all models in training run")
plt.plot(X,rho_acc.mean(0), label="PAC-Bayes ensemble, all models in training run")
plt.plot(X,best_acc.mean(0), label="Uniform ensemble, best model per training run")
plt.xlabel("Training runs")
plt.ylabel("Accuracy")
plt.title("Ensembles IMDB")
plt.legend()
plt.savefig("new_ens.png")
plt.close()
