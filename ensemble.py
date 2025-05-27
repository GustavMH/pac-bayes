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
from joblib import delayed, Parallel


def nan_argmax(arr):
    # NOTE np.argmax will return 0 for [nan,nan]
    res = np.full(arr.shape[:-1], np.nan)
    idxs = ~np.isnan(arr).any(axis=-1)
    argmax = np.argmax(arr, axis=-1)
    res[idxs] = argmax[idxs]
    return res

def nan_acc(X, y):
    idxs = ~np.isnan(X)
    return np.mean(X[idxs] == y[idxs])

def chkpnt_acc(chkpnt, val_labels):
    idxs = ~np.isnan(chkpnt).any(axis=-1)
    return np.mean(chkpnt[idxs].argmax(-1) == val_labels[idxs])


def ensemble_predictions_AVG(rho, predictions):
    pred = (predictions * rho[:, None, None]).sum(axis=0)
    pred = nan_argmax(pred)
    return pred


def make_rho_ensemble_oob(subset_idx, dataset):
    preds_test = dataset["predictions_test"][*subset_idx.T]
    labels_test = dataset["labels_test"]

    preds_val = dataset["predictions_validation"][*subset_idx.T]
    preds_val = nan_argmax(preds_val)
    labels_val = dataset["labels_validation"]

    bound, rho, lam = optimize_rho_oob(preds_val, labels_val)
    preds_ensemble_test = ensemble_predictions_AVG(rho, preds_test)

    return nan_acc(preds_ensemble_test, labels_test)


def make_uniform_ensemble(subset_idx, dataset):
    preds_test = dataset["predictions_test"][*subset_idx.T]
    labels_test = dataset["labels_test"]

    rho = np.ones(len(subset_idx)) / len(subset_idx)
    preds_ensemble_test = ensemble_predictions_AVG(rho, preds_test)

    return nan_acc(preds_ensemble_test, labels_test)

def make_early_stopping(subset_idx, dataset, val_scores = None):
    preds_val = dataset["predictions_validation"][*subset_idx.T]
    preds_test = dataset["predictions_test"][*subset_idx.T]
    labels_val = dataset["labels_validation"]
    labels_test = dataset["labels_test"]

    if isinstance(val_scores, np.ndarray):
        val_scores = np.array([chkpnt_acc(chkpnt, labels_val) for chkpnt in preds_val])

    preds_test = preds_test[val_scores.argmax()]
    return chkpnt_acc(preds_test, labels_test)

def make_rho_ensemble(subset_idx, dataset):
    preds_test = dataset["predictions_test"][*subset_idx.T]
    preds_val = dataset["predictions_validation"][*subset_idx.T]
    val_idx = ~np.isnan(preds_val[0, :, 0])
    preds_val = preds_val[:, val_idx, :].argmax(-1)
    labels_val = dataset["labels_validation"][val_idx]
    bound, rho, lam = optimize_rho(preds_val, labels_val)
    preds_ensemble_test = ensemble_predictions_AVG(rho, preds_test)
    return (preds_ensemble_test == dataset["labels_test"]).mean()

# TODO n_iter and n_checkpoints should come from the .npz file
for ds_name in [
        #("imdb", 50, 10),
        #("cifar10", 30, 15),
        #("cifar100", 30, 15),
        #("svhn", 30, 15)
        "Heart Disease"
]:
    pass

#ds_name = "Heart Disease"
ds_name = "Contraceptive Method Choice"

ds_npz = np.load(f"{ds_name}_predictions.npz")
ds = dict([(key, ds_npz[key]) for key in ds_npz.keys()])
n_iter, n_checkpoints, n_examples, n_cats = ds["predictions_validation"].shape
X = np.arange(2, n_checkpoints, step=1)

rho_acc = np.zeros((n_iter, len(X)))
uni_acc = np.zeros((n_iter, len(X)))
ear_acc = np.zeros((n_iter, len(X)))

val_scores = np.zeros((n_iter, n_checkpoints))
val_scores += np.array(
    Parallel(-1)(
        delayed(chkpnt_acc)(chkpnt, ds["labels_validation"])
        for chkpnt in ds["predictions_validation"] \
        .reshape(n_iter*n_checkpoints, n_examples, n_cats)
    )
).reshape(n_iter, n_checkpoints)

n_r = 1
for trial_idx in tqdm(np.arange(n_iter)):
    #chkpnts = [np.unique(np.linspace(0,X[-1],i).round()).astype(np.int64) for i in X]
    for _ in range(n_r):
        chkpnts = [np.random.choice(n_checkpoints, i) for i in X]
        subsets = [np.array([(trial_idx,c) for c in cs]) for cs in chkpnts]

        rho_acc[trial_idx] += np.array(Parallel(-1)(delayed(make_rho_ensemble_oob)(subset, ds) for subset in subsets))
        uni_acc[trial_idx] += np.array(Parallel(-1)(delayed(make_uniform_ensemble)(subset, ds) for subset in subsets))
        ear_acc[trial_idx] += np.array(Parallel(-1)(delayed(make_early_stopping)(subset, ds, val_scores) for subset in subsets))
    rho_acc[trial_idx] /= n_r
    uni_acc[trial_idx] /= n_r
    ear_acc[trial_idx] /= n_r

fig, axs = plt.subplots()
#np.savez(f"{ds_name}_graph_same_run.npz", rho_acc=rho_acc, uni_acc=uni_acc, ear_acc=ear_acc)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

grades = np.linspace(0,.1,5)
i = 0.25
uni_µ = np.mean(uni_acc, axis=0)
uni_std = np.std(uni_acc, axis=0)
plt.plot(X, uni_µ, label="Uniform ensemble of subset", color=colors[0])
#plt.fill_between(X, np.quantile(uni_acc, i, axis=0), np.quantile(uni_acc, 1-i, axis=0), alpha=0.1, color=colors[0])
plt.fill_between(X, uni_µ + uni_std, uni_µ - uni_std, color=colors[0], alpha=.1)

rho_µ = np.mean(rho_acc, axis=0)
rho_std = np.std(rho_acc, axis=0)
plt.plot(X, np.mean(rho_acc, axis=0), label="PAC-Bayes ensemble of subset", color=colors[1])
#plt.fill_between(X, np.quantile(rho_acc, i, axis=0), np.quantile(rho_acc, 1-i, axis=0), alpha=0.1, color=colors[1], linewidth=0)
plt.fill_between(X, rho_µ + rho_std, rho_µ - rho_std, color=colors[1], alpha=.1)

ear_µ = np.mean(ear_acc, axis=0)
ear_std = np.std(ear_acc, axis=0)
plt.plot(X, np.mean(ear_acc, axis=0), label="Best checkpoint of subset", color=colors[2])
#plt.fill_between(X, np.quantile(ear_acc, i, axis=0), np.quantile(ear_acc, 1-i, axis=0), alpha=0.1, color=colors[2], linewidth=0)
plt.fill_between(X, ear_µ + ear_std, ear_µ - ear_std, color=colors[2], alpha=.1)

plt.xlabel("Subset size ($N$ Checkpoints)")
plt.ylabel("Accuracy difference")
#plt.ylim(.76, .89)
plt.title(f"Ensembles {ds_name}")
plt.legend()
plt.savefig(f"{ds_name}_ens_same_run.png")
plt.close()


r_u = rho_acc - uni_acc
r_u_µ = np.mean(r_u, axis=0)
r_u_std = np.std(r_u, axis=0)
plt.plot(X, r_u_µ, label="PAC-Bayes minus Uniform ensemble", color=colors[0])
plt.fill_between(X, r_u_µ + r_u_std, r_u_µ - r_u_std, color=colors[0], alpha=.1)

r_e = rho_acc - ear_acc
r_e_µ = np.mean(r_e, axis=0)
r_e_std = np.std(r_e, axis=0)
plt.plot(X, r_e_µ, label="PAC-Bayes minus early stopping", color=colors[1])
plt.fill_between(X, r_e_µ + r_e_std, r_e_µ - r_e_std, color=colors[1], alpha=.1)

plt.xlabel("Subset size ($N$ Checkpoints)")
plt.ylabel("Accuracy difference")
plt.title(f"Ensembles {ds_name}")
plt.legend()
plt.savefig(f"{ds_name}_ens_diff.png")
plt.close()

fig, axs = plt.subplots(3,5, sharex=True, sharey=True, figsize=(8,5), layout="tight")

for ax, uni, rho, ear in zip(axs.flat, uni_acc, rho_acc, ear_acc):
    ax.plot(X, uni, label="Uniform ensemble of subset")
    ax.plot(X, rho, label="PAC-Bayes ensemble of subset")
    ax.plot(X, ear, label="Early stopping on subset")

fig.supxlabel("Subset size ($N$ Checkpoints)")
fig.supylabel("Accuracy")
plt.ylim(.6, .85)
fig.suptitle(f"Ensembles {ds_name}")
plt.legend()
plt.savefig(f"{ds_name}_ens_subplots.png")
plt.close()
