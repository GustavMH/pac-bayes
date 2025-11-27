#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.utils import check_random_state
from tqdm import tqdm
from joblib import Parallel, delayed
import time

from mvb import data as datasets
from mvb.bounds import optimizeLamb, optimizeCCTND, optimizeTND, optimizeBennett, bennett
from util import oob_tandem_risks, oob_gibbs_risks, split_bootstrap

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
        sample_idx, oob_idx = split_bootstrap(rng, X_train, Y_train, len(X_train), n_classes)
        estimator.fit(X_train[sample_idx], Y_train[sample_idx])
        Y_pred_test = estimator.predict(X_test)
        Y_pred_oob = estimator.predict(X_train[oob_idx])
        return Y_pred_test, Y_pred_oob, oob_idx

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


def gen_rf_risks(
        n_estimators = 100,
        n_iterations = 10,
        dataset_path = "/home/gustav/opgaver/ku/oracle-bounds/MajorityVoteBounds/NeurIPS2021/data/"
):
    def _run_estimators(X, Y, seed: int):
        print(f"run w. seed {seed}")
        start = time.time()
        rng = check_random_state(seed+1000)

        estimators = [
            Tree(
                criterion="gini",
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=rng,
                max_features="sqrt"
            ) for i in range(n_estimators)
        ]

        ds = datasets.split(X, Y, 0.8, random_state=rng)
        res = ensemble_predict_once(rng, estimators, *ds)
        print(f"_run_estimator took {time.time() - start}")
        return res

    def _run_iteration(dataset: str):
        print(dataset)
        X, Y = datasets.load(dataset, path=dataset_path)

        f = lambda i: _run_estimators(X, Y, i)
        res = Parallel(-1, backend="loky")(delayed(f)(i) for i in range(n_iterations))
        return np.array(res)

    # The Protein dataset has been removed from the datasets
    # where random forests are using in the original code
    ds_names = [
        "SVMGuide1", "Phishing", "Mushroom",
        "Splice", "w1a", "Cod-RNA", "Adult",
        "Connect-4", "Shuttle",
        "Pendigits", "Letter", "SatImage",
        "Sensorless", "USPS", "MNIST",
        "Fashion-MNIST"
    ]
    f = lambda name: (name, _run_iteration(name))
    res = (f(name) for name in ds_names)
    np.savez_compressed("random_forest.npz", **dict(res))


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
