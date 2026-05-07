#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from models.util import tandem_risks, gibbs_risks
import bounds
from itertools import product
from tqdm import tqdm

def load_eurosat():
    path = Path("~/Downloads/eurosat_chroma_shift.npz").expanduser()
    res = dict(np.load(path))

    # (10 runs, 5 training mixes, 2 test sets, 15 snapshots, 2700 examples, 10 categories)
    # Take model trained on A, see performance on A, trained for 10 epochs

    collect = [[[None] * 30] * 2] * 5
    for AB in [1,0]:
        val_labels = res["val_labels"][AB]
        test_labels = res["test_labels"][AB]
        for shift, i in tqdm(list(product(range(5), range(30)))):
            val_preds = res["validation"][:,shift,AB,i].argmax(-1)
            test_preds = res["test"][:,shift,AB,i].argmax(-1)

            collect[shift][AB][i] = calc_stats(val_preds, val_labels, test_preds, test_labels)

    return collect

try:
    res
except NameError:
    path = "~/Downloads/pac-bayes-predictions/imdb_predictions.npz"
    path = Path(path).expanduser()
    res = dict(np.load(path))

    collect = [None] * 50
    val_labels = res["labels_validation"]
    test_labels = res["labels_test"]
    for i in tqdm(range(50)):
        val_preds = res["predictions_validation"][i].argmax(-1)
        test_preds = res["predictions_test"][i].argmax(-1)

        collect[i] = calc_stats(val_preds, val_labels, test_preds, test_labels)

def vote(preds: np.array, rho: np.array):
    """vote weighted by RHO on PREDS"""
    return np.argmax(preds * rho, 0)

from scipy.special import softmax

def loss(rho, X, y):
    votes = (rho[:,None,None] * X).sum(0).argmax(-1)
    return 1-(votes == y).mean()

def est_feasible_region(X, y, n_iter=1000):
    simplex = np.random.randint(0,11,size=(1000,10))
    simplex = simplex / simplex.sum(-1)[:, None]
    opt = np.array([loss(rho, X, y) for rho in np.concat((simplex, np.eye(10)))])
    return np.max(opt), np.min(opt)

def max_feasible_region(X, y):
    hi = 1-(X == y[None, :]).any(0).mean()
    lo = 1-(X == y[None, :]).all(0).mean()
    return hi, lo

def calc_stats(val_preds, val_labels, test_preds, test_labels):
    risks, n1 = gibbs_risks(val_preds, val_labels)
    tnd, n2 = tandem_risks(val_preds, val_labels)
    params = {"tandem_risks": tnd, "n2": n2, "gibbs_risks": risks, "n1": n1}
    rho, bound, _ = bounds.optimize_rho("tnd", params)
    rho_fo, bound_fo, _ = bounds.optimize_rho("lambda", params)

    return {
        **params,
        "tnd_rho": rho,
        "tnd_bound": bound,
        "tnd_test_loss": loss(rho, np.eye(10)[test_preds], test_labels),
        "fo_rho": rho_fo,
        "fo_bound": bound_fo,
        "fo_test_loss": loss(rho_fo, np.eye(10)[test_preds], test_labels),
        "uni_test_loss": loss(np.ones(10) / 10, np.eye(10)[test_preds], test_labels),
        "uni_bound": loss(np.ones(10) / 10, np.eye(10)[val_preds], val_labels) + np.sqrt(np.log(2/0.05)/(2*len(val_preds[0]))),
        "min_feasible_region": est_feasible_region(np.eye(10)[test_preds], test_labels),
        "max_feasible_region": max_feasible_region(test_preds, test_labels)
    }


def plot_mats(mats, titles=[], figname="risks"):
    fig, axss = plt.subplots(len(mats),1,figsize=(5.5,5),sharex=True,sharey=True,layout="compressed")

    fig.supxlabel("Epoch")
    fig.supylabel("Training run")

    idx = np.argsort(mats[-1][-1])
    for ax, mat, title in zip(axss, mats, titles):
        ax.set_title(title)
        m = ax.imshow(mat.T[idx], interpolation='nearest', aspect='auto')

    plt.savefig(f"fig/{figname}.png")
    plt.close()


def plot_test_perf(risks, min_region, max_region, risk_labels=[], figname="test_perf"):
    fig, ax = plt.subplots(1,1,figsize=(5.5,4),sharex=True,sharey=True,layout="compressed")
    plt.title(f"Voting ensembles, 10 members, IMDB")

    for risk, label in zip(risks, risk_labels):
        ax.plot(risk, label=label)

    #ax.fill_between(np.arange(len(min_region[:,0])), min_region[:,0], min_region[:,1], label=f"Min. feasible region", alpha=0.2)
    #ax.plot(max_region[:,0], label=f"Max. feasible region", alpha=0.5, c="tab:blue", linestyle="dotted")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.legend()

    plt.savefig(f"fig/{figname}.png")
    plt.close()


def plot_bounds():
    fig, ax = plt.subplots(1,1,figsize=(5.5,4),sharex=True,sharey=True,layout="tight")
    plt.title("Voting ensembles, uniform, 10 members")
    ax.plot(collect_uni_A, label="Uniform loss A")
    ax.fill_between(np.arange(30), collect_uni_A, collect_bounds_uni_A, label="Loss bound A", alpha=0.2)

    ax.plot(collect_uni_B, label="Uniform loss B")
    ax.fill_between(np.arange(30), collect_uni_B, collect_bounds_uni_B, label="Loss bound B", alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend()

    plt.savefig(f"fig/{figname}.png")
    plt.close()


def plot_voting():
    X=softmax(res["test"][:,0,0,0])
    y=res["test_labels"][0]

    all_idx = ~(Xm == Xm[0]).all(0)
    Xs, ys = X[:, all_idx], y[all_idx]

    loss(np.ones(10)/10, Xs, ys)

    ind = [i for i, _ in sorted(enumerate(np.concat([ys[None,:],Xs.argmax(-1)]).T), key=lambda x: tuple(x[1]))]
    ind = np.array(ind)

    fig, ax = plt.subplots(1,1,figsize=(5.5,2),sharex=True,sharey=True,layout="tight")
    ax.imshow(np.take_along_axis(Xs.argmax(-1).T, ind[:,None], axis=0).T, cmap="tab10", interpolation="nearest", aspect="auto")
    plt.savefig("fig/votes.png")
    plt.close()
