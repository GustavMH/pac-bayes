#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from models.util import tandem_risks, gibbs_risks, oob_tandem_risks, oob_gibbs_risks
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

def load_IMDB(path = "~/Downloads/pac-bayes-predictions/imdb_predictions.npz"):
    path = Path(path).expanduser()
    res = dict(np.load(path))

    collect = [None] * 50
    val_labels = res["labels_validation"]
    test_labels = res["labels_test"]
    for i in tqdm(range(50)):
        val_preds = res["predictions_validation"][i].argmax(-1)
        test_preds = res["predictions_test"][i].argmax(-1)

        collect[i] = calc_stats(val_preds, val_labels, test_preds, test_labels)

    return collect

def load_CIFAR10(path = "~/Downloads/pac-bayes-predictions/cifar10_predictions.npz"):
    path = Path(path).expanduser()
    res = dict(np.load(path))

    collect = [None] * 30
    val_labels = res["labels_validation"]
    val_index = np.isnan(res["predictions_validation"][:,0,:,0])
    val_index = np.array([np.arange(len(nans))[~nans] for nans in val_index])
    test_labels = res["labels_test"]
    for i, idx in tqdm(enumerate(val_index)):
        val_preds = res["predictions_validation"][i, :, idx].argmax(-1).T
        test_preds = res["predictions_test"][i].argmax(-1)

        collect[i] = calc_stats(val_preds, val_labels[idx], test_preds, test_labels)

    return collect

def fix_CIFAR10():
    path = "~/Downloads/pac-bayes-predictions/cifar10_predictions.npz"
    path = Path(path).expanduser()
    res = dict(np.load(path))

    val_labels = res["labels_validation"]
    val_preds = res["predictions_validation"]
    val_index = np.isnan(val_preds[:,0,:,0])
    val_index = np.array([np.arange(len(nans))[~nans] for nans in val_index])
    val_labels_sub = val_labels[val_index]
    val_preds_sub = np.array([val_preds[i,:,val_index[i]] for i in range(len(val_index))]).transpose((0,2,1,3))

    return collect

def load_CIFAR100(path = "~/Downloads/pac-bayes-predictions/cifar100_predictions.npz"):
    path = Path(path).expanduser()
    res = np.load(path, mmap_mode="r")

    collect = [None] * 30
    val_labels = res["labels_validation"]
    val_index = np.isnan(res["predictions_validation"][:,0,:,0])
    val_index = np.array([np.arange(len(nans))[~nans] for nans in val_index])
    test_labels = res["labels_test"]
    for i, idx in tqdm(enumerate(val_index)):
        val_preds = res["predictions_validation"][i, :, idx].argmax(-1).T
        test_preds = res["predictions_test"][i].argmax(-1)

        collect[i] = calc_stats(val_preds, val_labels[idx], test_preds, test_labels)

def vote(preds: np.array, rho: np.array):
    """vote weighted by RHO on PREDS"""
    return np.argmax(preds * rho, 0)

from scipy.special import softmax

def loss(rho, X, y):
    votes = (rho[:,None,None] * X).sum(0).argmax(-1)
    return 1-(votes == y).mean()

def est_feasible_region(X, y, n_iter=1000):
    n_models, *_ = X.shape
    simplex = np.random.randint(0,n_models+1,size=(1000,n_models))
    simplex = simplex / simplex.sum(-1)[:, None]
    opt = np.array([loss(rho, X, y) for rho in np.concat((simplex, np.eye(n_models)))])
    return np.max(opt), np.min(opt)

def max_feasible_region(X, y):
    hi = 1-(X == y[None, :]).any(0).mean()
    lo = 1-(X == y[None, :]).all(0).mean()
    return hi, lo

def calc_stats(val_preds, val_labels, test_preds, test_labels, n_iter_region=1000):
    n_models, n_examples = val_preds.shape
    n_cats = 1+val_labels.max()
    uni = np.ones(n_models) / n_models

    risks, n1 = gibbs_risks(val_preds, val_labels)
    tnd, n2 = tandem_risks(val_preds, val_labels)
    params = {"tandem_risks": tnd, "n2": n2, "gibbs_risks": risks, "n1": n1}
    rho, bound, _ = bounds.optimize_rho("tnd", params)
    rho_fo, bound_fo, _ = bounds.optimize_rho("lambda", params)

    return {
        **params,
        "tnd_rho": rho,
        "tnd_bound": bound,
        "tnd_test_loss": loss(rho, np.eye(n_cats)[test_preds], test_labels),
        "fo_rho": rho_fo,
        "fo_bound": bound_fo,
        "fo_test_loss": loss(rho_fo, np.eye(n_cats)[test_preds], test_labels),
        "uni_test_loss": loss(uni, np.eye(n_cats)[test_preds], test_labels),
        "uni_bound": loss(uni, np.eye(n_cats)[val_preds], val_labels) + np.sqrt(np.log(2/0.05)/(2*len(val_preds[0]))),
        "min_feasible_region": est_feasible_region(np.eye(n_cats)[test_preds], test_labels, n_iter=n_iter_region),
        "max_feasible_region": max_feasible_region(test_preds, test_labels)
    }

def calc_oob_stats(val_preds, val_idx, val_labels, test_preds, test_labels, n_iter_region=1000):
    n_models, n_examples = val_preds.shape
    n_cats = 1+val_labels.max()
    uni = np.ones(n_models) / n_models

    risks, n1 = oob_gibbs_risks(val_preds, val_idx, val_labels)
    tnd, n2 = oob_tandem_risks(val_preds, val_idx, val_labels)
    params = {"tandem_risks": tnd, "n2": n2, "gibbs_risks": risks, "n1": n1}
    rho, bound, _ = bounds.optimize_rho("tnd", params)
    rho_fo, bound_fo, _ = bounds.optimize_rho("lambda", params)

    return {
        **params,
        "tnd_rho": rho,
        "tnd_bound": bound,
        "tnd_test_loss": loss(rho, np.eye(n_cats)[test_preds], test_labels),
        "fo_rho": rho_fo,
        "fo_bound": bound_fo,
        "fo_test_loss": loss(rho_fo, np.eye(n_cats)[test_preds], test_labels),
        "uni_test_loss": loss(uni, np.eye(n_cats)[test_preds], test_labels),
        "uni_bound": loss(uni, np.eye(n_cats)[val_preds], val_labels) + np.sqrt(np.log(2/0.05)/(2*len(val_preds[0]))),
    }


def plot_mats(mats, titles=[], figname="risks"):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    fig, axss = plt.subplots(1,len(mats),figsize=(5.5,1.5),sharex=True,sharey=True,layout="compressed")

    fig.supxlabel("Epoch")
    fig.supylabel("Training run")

    idx = np.argsort(mats[-1][-1])
    for ax, mat, title in zip(axss, mats, titles):
        ax.set_title(title)
        ax.set_yticks([])
        m = ax.imshow(mat.T[idx], aspect='auto')

    plt.savefig(f"fig/{figname}.pdf")
    plt.close()


def plot_test_perf(risks, min_region, max_region, risk_labels=[], figname="test_perf"):
    fig, ax = plt.subplots(1,1,figsize=(5.5,2.5),sharex=True,sharey=True,layout="compressed")
    plt.title(f"Voting ensembles, 10 members, IMDB")

    for risk, label in zip(risks, risk_labels):
        ax.plot(risk, label=label)

    #ax.fill_between(np.arange(len(min_region[:,0])), min_region[:,0], min_region[:,1], label=f"Min. feasible region", alpha=0.2)
    #ax.plot(max_region[:,0], label=f"Max. feasible region", alpha=0.5, c="tab:blue", linestyle="dotted")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend()

    plt.savefig(f"fig/{figname}.pdf")
    plt.close()


def plot_test_perf_imdb():
    imdb = load_IMDB()

    risks = [a["uni_test_loss"] for a in imdb]
    rho_fo = [a["fo_test_loss"] for a in imdb]
    rho_tnd = [a["tnd_test_loss"] for a in imdb]

    plot_test_perf(
        np.array([rho_fo,risks,   rho_tnd]),
        None, None,
        ["First-order","Uniform",   "Tandem"]
    )

def plot_test_perf_scatter_cifar10():
    from scipy.stats import mannwhitneyu as utest

    cifar = load_CIFAR10()
    uni = np.array([a["uni_test_loss"] for a in cifar])
    fo = np.array([a["fo_test_loss"] for a in cifar])
    tnd = np.array([a["tnd_test_loss"] for a in cifar])

    fig, ax = plt.subplots(1,2,figsize=(5.5,2),layout="compressed",width_ratios=[3,2])
    plt.suptitle(f"Voting ensembles, 15 members, CIFAR10")

    scatter = .1*rng.uniform(-1,1,size=len(uni))
    for risk, label, i in zip([fo,uni,tnd], ["First-order", "Uniform", "Tandem"], [0,1,2]):
        ax[0].plot(i * np.ones_like(risk) + scatter, risk, label=label, marker="o", linestyle="None", alpha=.5)
    ax[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax[0].set_xlim(-.5,2.5)
    ax[0].set_xticks([0,1,2], ["First-order", "Uniform", "Tandem"])
    ax[0].set_ylabel("Loss")


    for risk, label, i, c in zip([fo-uni,tnd-uni], ["First-order", "Tandem"], [0,1,2], ["tab:blue", "tab:green"]):
        ax[1].plot(i * np.ones_like(risk) + scatter, risk, label=label, marker="o", linestyle="None", alpha=.5, c=c)

    ax[1].set_xlim(-.5,1.5)
    ax[1].set_xticks([0,1], ["First-order", "Tandem"])
    ax[1].set_ylabel("Loss change \n (PAC-Bayes - Uniform)")
    ax[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=1, symbol="pp"))

    plt.savefig(f"fig/cifar_perf.pdf")
    plt.close()

    print("uniform vs. fo", utest(uni, fo))
    print("uniform vs. tnd", utest(uni, tnd))

def plot_test_perf_scatter_imdb():
    from scipy.stats import mannwhitneyu as utest

    imdb = load_IMDB()

    uni = np.array([a["uni_test_loss"] for a in imdb])
    fo  = np.array([a["fo_test_loss"] for a in imdb])
    tnd = np.array([a["tnd_test_loss"] for a in imdb])

    fig, ax = plt.subplots(1,2,figsize=(5.5,2),layout="compressed",width_ratios=[3,2])
    plt.suptitle(f"Voting ensembles, 10 members, IMDB")

    scatter = .1*rng.uniform(-1,1,size=len(uni))
    for risk, label, i in zip([fo,uni,tnd], ["First-order", "Uniform", "Tandem"], [0,1,2]):
        ax[0].plot(i * np.ones_like(risk) + scatter, risk, label=label, marker="o", linestyle="None", alpha=.5)
    ax[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax[0].set_xlim(-.5,2.5)
    ax[0].set_xticks([0,1,2], ["First-order", "Uniform", "Tandem"])
    ax[0].set_ylabel("Loss")


    for risk, label, i, c in zip([fo-uni,tnd-uni], ["First-order", "Tandem"], [0,1,2], ["tab:blue", "tab:green"]):
        ax[1].plot(i * np.ones_like(risk) + scatter, risk, label=label, marker="o", linestyle="None", alpha=.5, c=c)

    ax[1].set_xlim(-.5,1.5)
    ax[1].set_xticks([0,1], ["First-order", "Tandem"])
    ax[1].set_ylabel("Loss change \n (PAC-Bayes - Uniform)")
    ax[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol="pp"))

    plt.savefig(f"fig/imdb_perf.pdf")
    plt.close()

    print("uniform vs. fo", utest(uni, fo))
    print("uniform vs. tnd", utest(uni, tnd))
    print("fo vs. tnd", utest(fo, tnd))

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


def load_IMDB_strata():
    path = "~/Downloads/pac-bayes-predictions/imdb_predictions.npz"
    path = Path(path).expanduser()
    res = dict(np.load(path))

    collect = []
    val_labels = res["labels_validation"]
    test_labels = res["labels_test"]
    for i, n_models in tqdm(list(product(range(5),range(2,11)))):
        val_preds = res["predictions_validation"][i, :n_models].argmax(-1)
        test_preds = res["predictions_test"][i, :n_models].argmax(-1)

        collect.append(calc_stats(val_preds, val_labels, test_preds, test_labels, n_iter_region=1))

    return np.array(collect).reshape(5,9)

def load_IMDB_bootstrap(rng):
    path = "~/Downloads/pac-bayes-predictions/imdb_predictions.npz"
    path = Path(path).expanduser()
    res = dict(np.load(path))

    collect = []
    val_labels = res["labels_validation"]
    test_labels = res["labels_test"]
    for i, n_models in tqdm(list(product(range(10),range(2,30)))):
        idx = rng.integers(0,500, size=n_models)
        val_preds = res["predictions_validation"].reshape((500,25000,2))[idx].argmax(-1)
        test_preds = res["predictions_test"].reshape((500,25000,2))[idx].argmax(-1)

        collect.append(calc_stats(val_preds, val_labels, test_preds, test_labels, n_iter_region=1))

    return np.array(collect).reshape(10,28)

rng = np.random.default_rng()

def plot_IMDB_bootstrap(rng):
    collect = load_IMDB_bootstrap(rng)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    fig, ax = plt.subplots(1,1,figsize=(5.5,3),layout="compressed")
    plt.title("IMDB ensembles, bootstraping ensemble members")

    µ_fo = np.array([np.array([r["fo_test_loss"] for r in collect[:,i]]).mean() for i in range(28)])
    µ_uni = np.array([np.array([r["uni_test_loss"] for r in collect[:,i]]).mean() for i in range(28)])
    µ_tnd = np.array([np.array([r["tnd_test_loss"] for r in collect[:,i]]).mean() for i in range(28)])
    std_fo = np.array([np.array([r["fo_test_loss"] for r in collect[:,i]]).std() for i in range(28)])
    std_uni = np.array([np.array([r["uni_test_loss"] for r in collect[:,i]]).std() for i in range(28)])
    std_tnd = np.array([np.array([r["tnd_test_loss"] for r in collect[:,i]]).std() for i in range(28)])

    ax.plot(µ_fo, label="First-order")
    ax.plot(µ_uni, label="Uniform")
    ax.plot(µ_tnd, label="Tandem")
    ax.set_ylim(bottom=0.12, top=0.25)
    ax.fill_between(np.arange(len(µ_fo)), µ_fo+std_fo, µ_fo-std_fo, alpha=0.2)
    ax.fill_between(np.arange(len(µ_uni)), µ_uni+std_uni, µ_uni-std_uni, alpha=0.2)
    ax.fill_between(np.arange(len(µ_tnd)), µ_tnd+std_tnd, µ_tnd-std_tnd, alpha=0.2)
    ax.set_ylabel("loss")
    ax.set_xlabel("No. of ensemble members")
    plt.legend()
    plt.savefig("fig/IMDB_bootstrap.pdf")
    plt.close()

def plot_imdb_risks_v_weights():
    imdb = load_IMDB()

    risks = [a["gibbs_risks"] for a in imdb]
    rho_fo = [a["fo_rho"] for a in imdb]
    rho_tnd = [a["tnd_rho"] for a in imdb]

    plot_mats(
        np.array([risks, rho_fo, rho_tnd]),
        ["Loss",  "First-order weights", "Tandem weights"]
    )

def plot_imdb_strata():
    #collect = load_IMDB_strata()

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    fig, ax = plt.subplots(1,1,figsize=(5.5,3),layout="compressed")
    plt.title("IMDB ensembles, bootstraping ensemble members")

    µ_fo = np.array([np.array([r["fo_test_loss"] for r in collect[:,i]]).mean() for i in range(9)])
    µ_uni = np.array([np.array([r["uni_test_loss"] for r in collect[:,i]]).mean() for i in range(9)])
    µ_tnd = np.array([np.array([r["tnd_test_loss"] for r in collect[:,i]]).mean() for i in range(9)])
    std_fo = np.array([np.array([r["fo_test_loss"] for r in collect[:,i]]).std() for i in range(9)])
    std_uni = np.array([np.array([r["uni_test_loss"] for r in collect[:,i]]).std() for i in range(9)])
    std_tnd = np.array([np.array([r["tnd_test_loss"] for r in collect[:,i]]).std() for i in range(9)])

    ax.plot(µ_fo, label="First-order")
    ax.plot(µ_uni, label="Uniform")
    ax.plot(µ_tnd, label="Tandem")
    ax.fill_between(np.arange(len(µ_fo)), µ_fo+std_fo, µ_fo-std_fo, alpha=0.2)
    ax.fill_between(np.arange(len(µ_uni)), µ_uni+std_uni, µ_uni-std_uni, alpha=0.2)
    ax.fill_between(np.arange(len(µ_tnd)), µ_tnd+std_tnd, µ_tnd-std_tnd, alpha=0.2)
    ax.set_ylabel("loss")
    ax.set_xlabel("No. of bad ensemble members")
    plt.legend()
    plt.savefig("fig/IMDB_strata.pdf")
    plt.close()
