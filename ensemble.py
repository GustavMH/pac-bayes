#!/usr/bin/env python3
import re
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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


def make_rho_ensemble_oob(subset_idx, dataset):
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

def plot_bootstrap():
    imdb = np.load("imdb_predictions.npz")
    idxs = np.array(select_models(imdb, "all"))
    idxs_best = np.array(select_models(imdb, "best"))

    n_iter = 5
    X = np.arange(2,21)

    rho_acc = np.zeros((n_iter, len(X)))
    uni_acc = np.zeros((n_iter, len(X)))
    best_acc = np.zeros((n_iter, len(X)))

    for i in range(n_iter):
        print(i)
        subsets_models = [np.random.choice(np.arange(50),i)for i in X]
        subsets_chkpnt = [np.array([(i,j) for i in s for j in np.arange(10)]) for s in subsets_models]

        rho_acc[i] = np.array([make_rho_ensemble(subset, imdb) for subset in subsets_chkpnt])
        uni_acc[i] = np.array([make_uniform_ensemble(subset, imdb) for subset in subsets_chkpnt])
        best_acc[i] = np.array([make_early_stopping(idxs_best[subset], imdb) for subset in subsets_models])

    plt.plot(X,uni_acc.mean(0), label="Uniform ensemble, all models in training run")
    plt.plot(X,rho_acc.mean(0), label="PAC-Bayes ensemble, all models in training run")
    plt.plot(X,best_acc.mean(0), label="Uniform ensemble, best model per training run")
    plt.xlabel("Training runs")
    plt.ylabel("Accuracy")
    plt.title(f"Ensembles {ds_name}")
    plt.legend()
    plt.savefig(f"{ds_name}_ens.png")
    plt.close()

    for ds_name in ["cifar10", "cifar100", "svhn"]:
        ds = np.load(f"{ds_name}_predictions.npz")
        idxs = np.array(select_models(ds, "all"))
        idxs_best = np.array(select_models(ds, "best"))

        n_iter = 5
        X = np.arange(2,21)

        rho_acc = np.zeros((n_iter, len(X)))
        uni_acc = np.zeros((n_iter, len(X)))
        best_acc = np.zeros((n_iter, len(X)))

        for i in range(n_iter):
            print(i)
            subsets_models = [np.random.choice(np.arange(30),i) for i in X]
            subsets_chkpnt = [np.array([(i,j) for i in s for j in np.arange(10)]) for s in subsets_models]

            rho_acc[i] = np.array([make_rho_ensemble(subset, ds) for subset in subsets_chkpnt])
            uni_acc[i] = np.array([make_uniform_ensemble(subset, ds) for subset in subsets_chkpnt])
            best_acc[i] = np.array([make_early_stopping(idxs_best[subset], imdb) for subset in subsets_models])
            #best_acc[i] = np.array([make_rho_ensemble(idxs_best[subset], ds) for subset in subsets_models])

        np.savez(f"{ds_name}_graph.npz" )

        plt.plot(X,uni_acc.mean(0), label="Uniform ensemble, all models in training run")
        plt.plot(X,rho_acc.mean(0), label="PAC-Bayes ensemble, all models in training run")
        plt.plot(X,best_acc.mean(0), label="Uniform ensemble, best model per training run")
        plt.xlabel("Training runs")
        plt.ylabel("Accuracy")
        plt.title(f"Ensembles {ds_name}")
        plt.legend()
        plt.savefig(f"{ds_name}_ens.png")
        plt.close()

def make_rho_ensemble(subset_idx, dataset):
    preds_test = dataset["predictions_test"][*subset_idx.T]
    preds_val = dataset["predictions_validation"][*subset_idx.T]
    val_idx = ~np.isnan(preds_val[0, :, 0])
    preds_val = preds_val[:, val_idx, :].argmax(-1)
    labels_val = dataset["labels_validation"][val_idx]
    bound, rho, lam = optimize_rho(preds_val, labels_val)
    preds_ensemble_test = ensemble_predictions_AVG(rho, preds_test)
    return (preds_ensemble_test == dataset["labels_test"]).mean()

for ds_name, n_iter, n_checkpoints in [
        ("imdb", 50, 10),
        ("cifar10", 30, 15),
        ("cifar100", 30, 15),
        ("svhn", 30, 15)
]:
    ds_npz = np.load(f"{ds_name}_predictions.npz")
    ds = dict([(key, ds_npz[key]) for key in ds_npz.keys()])

    #idxs = np.array(select_models(ds, "all"))
    X = np.arange(2,n_checkpoints+1)

    rho_acc = np.zeros((n_iter, len(X)))
    uni_acc = np.zeros((n_iter, len(X)))
    ear_acc = np.zeros((n_iter, len(X)))

    for trail_idx in tqdm(np.arange(n_iter)):
        chkpnts = [np.unique(np.linspace(0,len(X),i).round()).astype(np.int64) for i in X]
        subsets = [np.array([(trail_idx,c) for c in cs]) for cs in chkpnts]

        rho_acc[trail_idx] = np.array([make_rho_ensemble(subset, ds) for subset in subsets])
        uni_acc[trail_idx] = np.array([make_uniform_ensemble(subset, ds) for subset in subsets])
        ear_acc[trail_idx] = np.array([make_early_stopping(subset, ds) for subset in subsets])

    np.savez(f"{ds_name}_graph_same_run.npz", rho_acc=rho_acc, uni_acc=uni_acc, ear_acc=ear_acc)

    plt.plot(X, np.median(uni_acc, axis=0), label="Uniform ensemble of subset")
    plt.fill_between(X, np.quantile(uni_acc, .25, axis=0), np.quantile(uni_acc, .75, axis=0), alpha=0.2)
    plt.plot(X, np.median(rho_acc, axis=0), label="PAC-Bayes ensemble of subset")
    plt.fill_between(X, np.quantile(rho_acc, .25, axis=0), np.quantile(rho_acc, .75, axis=0), alpha=0.2)
    plt.plot(X, np.median(ear_acc, axis=0), label="Best checkpoint")
    plt.fill_between(X, np.quantile(ear_acc, .25, axis=0), np.quantile(ear_acc, .75, axis=0), alpha=0.2)
    plt.xlabel("Checkpoints ($N$)")
    plt.ylabel("Accuracy")
    plt.title(f"Ensembles {ds_name}")
    plt.legend()
    plt.savefig(f"{ds_name}_ens_same_run.png")
    plt.close()
