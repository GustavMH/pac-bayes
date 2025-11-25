#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree
from itertools import islice
from tqdm import tqdm
from joblib import Parallel, delayed

from mvb import data as datasets
from mvb.bounds import optimizeLamb, optimizeCCTND, optimizeTND, optimizeBennett, bennett

def oob_tandem_risks(
        preds: [np.array],
        pred_idx: [np.array],
        targs: np.array
):
    """
    Calculate a (model, model) size array of risk scores.

    Predictions from an out-of-bag (unused for optimization) validation
    set; the PREDS array has shape (model, prediction), the TARGS array
    contains the dataset from which the validation set has been drawn
    """
    assert(len(preds) == len(pred_idx))
    assert(all(p.shape == i.shape for (p,i) in zip(preds, pred_idx)))

    m = len(preds)
    tandem_risks = np.zeros((m,m))
    n_intersects = np.zeros((m,m))

    # Upper triangular martix
    for i, (preds_a, idx_a) in enumerate(zip(preds, pred_idx)):
        for j, (preds_b, idx_b) in islice(enumerate(zip(preds, pred_idx)), 0, i):
            idxs, c_a, c_b = np.intersect1d(idx_a, idx_b, True, True)

            tandem_risks[i,j] = np.sum(
                np.logical_and(
                    preds_a[c_a] != targs[idxs],
                    preds_b[c_b] != targs[idxs]
                )
            )

            n_intersects[i,j] = len(idxs)

    # Lower triangular martix
    tandem_risks += tandem_risks.mT
    n_intersects += n_intersects.mT

    # Diagonal
    for i, (preds, idxs) in enumerate(zip(preds, pred_idx)):
        tandem_risks[i,i] = np.sum(preds != targs[idxs])
        n_intersects[i,i] = len(idxs)

    return tandem_risks / n_intersects, n_intersects.min()


def oob_gibbs_risks(preds, pred_idx, targs):
    """
    Calculate a (model) size array of risk scores.

    Predictions from an out-of-bag (unused for optimization) validation
    set; the PREDS array has shape (model, prediction), the TARGS array
    contains the dataset from which the validation set has been drawn
    """
    assert(len(preds) == len(pred_idx))
    assert(all(p.shape == i.shape for (p,i) in zip(preds, pred_idx)))

    m = len(preds)
    risks = np.zeros(m)
    n_intersects = np.zeros(m)

    for i, (preds, idxs) in enumerate(zip(preds, pred_idx)):
        risks[i] = np.sum(preds != targs[idxs])
        n_intersects[i] = len(idxs)

    return risks / n_intersects, n_intersects.min()


def split_bootstrap(
        rng: np.random.RandomState,
        X: np.array,
        Y: np.array,
        n_samples: int,
        n_classes: None | int
) -> ((np.array, np.array), (np.array, np.array)):
    """Sample points (x,y) in (X,Y) w. replacement, get at least one example per class in Y.
    Return the sample and sample indicies and out-of-bag samples and indicies"""
    n_classes = n_classes or len(np.unique(Y))

    X_sample = np.empty((n_samples, *X.shape[1:]))
    Y_sample = np.empty((n_samples, *Y.shape[1:]))

    while not len(np.unique(Y_sample)) == n_classes:
        idx = rng.integers(n_samples, size=n_samples)
        Y_sample = Y[idx]

    oob_idx = np.delete(np.arange(len(X)), idx)

    return (X[idx], Y_sample, idx), (X[oob_idx], Y[oob_idx], oob_idx)


def ensemble_predict_once(
        rng: np.random.RandomState,
        estimators,
        X_train: np.array,
        Y_train: np.array,
        X_test: np.array,
        Y_test: np.array
):
    n_classes = len(np.unique(Y_train))

    def pred(estimator):
        (X_train_, Y_train_, _), (X_oob, Y_oob, idx_oob) = \
            split_bootstrap(rng, X_train, Y_train, len(X_train), n_classes=n_classes)
        estimator.fit(X_train, Y_train)
        Y_pred_test = estimator.predict(X_test)
        Y_pred_oob = estimator.predict(X_oob)
        return Y_pred_test, Y_pred_oob, idx_oob

    Y_pred_test = np.empty((len(estimators), len(X_test)))
    Y_pred_oob  = [None] * len(estimators)
    Y_pred_idx  = [None] * len(estimators)

    for i, e in enumerate(estimators):
        Y_pred_test[i], Y_pred_oob[i], Y_pred_idx[i] = pred(e)

    L, n1 = oob_gibbs_risks(Y_pred_oob, Y_pred_idx, Y_train)
    L_tnd, n2 = oob_tandem_risks(Y_pred_oob, Y_pred_idx, Y_train)
    L_test, n_test = oob_gibbs_risks(
        Y_pred_test,
        [np.arange(len(Y_test))] * len(estimators),
        Y_test
    )

    return {
        "gibbs_risks": L,
        "n1": n1,
        "tandem_risks": L_tnd,
        "n2": n2,
        "gibbs_test_risks": L_test,
        "n_test": n_test
    }


def optimize_rho(bound: str, params: dict = {}):
    # Prior weights, used to calculate kl(pi||rho)
    L = params["gibbs_risks"]
    n1 = params["n1"]
    L_tnd = params["tandem_risks"]
    n2 = params["n2"]
    pi = np.ones(len(L)) / len(L)
    rho = None
    extra = {}

    match bound:
        case "best":
            rho = np.insert(np.zeros(len(L)-1), np.argmin(L), 1)
        case "uniform":
            rho = np.ones(len(L)) / len(L)
        case "lambda":
            (bound, rho, lam) = optimizeLamb(L, n1, abc_pi=pi)
            extra = { "lambda": lam }
        case "tnd":
            (bound, rho, lam) = optimizeTND(L_tnd, n2, abc_pi=pi)
            extra = { "lambda": lam }
        case "cctnd":
            (bound, rho, mu, lam, gamma) = optimizeCCTND(L_tnd, L, n1, n2, abc_pi=pi)
            extra = { "lambda": lam, "mu": mu, "gamma": gamma }
        case "bennett":
            (rho, alpha, beta, lam) = optimizeBennett(L_tnd, L, 1, 1, n1, n2, 1, pi)
            bound = bennett(L_tnd, L, alpha, beta, n1, n2, lam, pi, rho)

            extra = { "alpha": alpha, "beta": beta, "lambda": lam }
        case _:
            raise f"Unknown bound type: {bound}"

    return rho, bound, extra

def mv_loss(L, rho):
    return (rho*L).sum()

def gen_rf_risks(n_estimators = 100, n_iterations = 10):
    def _run_estimators(dataset: str, seed: int):
        estimators = [
            Tree(
                criterion="gini",
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=i+seed,
                max_features="sqrt"
            ) for i in range(n_estimators)
        ]

        rng = np.random.default_rng(1000+seed)
        X,Y = datasets.load(dataset, path="./MajorityVoteBounds/NeurIPS2021/data/")
        ds = datasets.split(X, Y, 0.8, random_state=100+seed)

        return ensemble_predict_once(rng, estimators, *ds)

    def _run_iteration(dataset: str):
        f = lambda i: _run_estimators(dataset, i)
        res = Parallel(-1)(delayed(f)(i) for i in range(n_iterations))
        return np.array(res)

    ds_names = [
        "SVMGuide1" , "Phishing", "Mushroom",
        "Splice", "w1a", "Cod-RNA", "Adult",
        "Protein", "Connect-4", "Shuttle",
        "Pendigits", "Letter", "SatImage",
        "Sensorless", "USPS", "MNIST",
        "Fashion-MNIST"
    ]
    f = lambda name: (name, _run_iteration(name))
    res = (f(name) for name in ds_names)
    res = dict(res)
    old = np.load("random_forest.npz", allow_pickle=True)
    np.savez_compressed("random_forest.npz", **old, **res)


def gen_bounds(f: Path):
    ds_npz = np.load(f, allow_pickle=True)
    ds = dict([(key, ds_npz[key]) for key in ds_npz.keys()])
    bounds = ["lambda", "tnd", "cctnd", "bennett"]

    def eval_bound(bound, dataset):
        res = [optimize_rho(bound, params) for params in ds[dataset]]
        bounds = [b for _, b, _ in res]
        rhos = [r for r, _, _ in res]
        mv = [(params["gibbs_test_risks"]*rho).sum()
              for params, rho in zip(ds[dataset], rhos)]
        return np.mean(bounds), np.mean(mv)

    return np.array([[
        eval_bound(bound, dataset)
        for dataset in ds.keys()
    ] for bound in bounds])
