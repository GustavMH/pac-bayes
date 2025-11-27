#!/usr/bin/env python3

import numpy as np
from pathlib import Path

from util import arr2d_to_df
from mvb.bounds import optimizeLamb, lamb, optimizeCCTND, CCTND, optimizeTND, TND, optimizeBennett, bennett
from mvb import util

def optimize_rho(bound: str, params: dict = {}):
    # Prior weights, used to calculate kl(pi||rho)
    L = params["gibbs_risks"]
    n1 = params["n1"]
    L_tnd = params["tandem_risks"]
    n2 = params["n2"]
    pi = np.ones(len(L)) / len(L)
    rho = None
    extra = {}

    def tandem_risk(rho):
        return (L_tnd*np.outer(rho,rho)).sum()

    def gibbs_risk(rho):
        return (L*rho).sum()

    match bound:
        case "best":
            rho = np.insert(np.zeros(len(L)-1), np.argmin(L), 1)
        case "uniform":
            rho = np.ones(len(L)) / len(L)
        case "lambda": # First order
            (_, rho, lam) = optimizeLamb(L, n1, abc_pi=pi)
            bound = lamb(gibbs_risk(rho), n1, util.kl(rho, pi))
            extra = { "lambda": lam }
        case "tnd":
            (_, rho, lam) = optimizeTND(L_tnd, n2, abc_pi=pi)
            bound = TND(tandem_risk(rho), n2, util.kl(rho, pi))
            extra = { "lambda": lam }
        case "cctnd":
            (bound, rho, mu, lam, gamma) = optimizeCCTND(L_tnd, L, n1, n2, abc_pi=pi)
            bound = CCTND(tandem_risk(rho), gibbs_risk(rho), n1, n2, util.kl(rho, pi))
            extra = { "lambda": lam, "mu": mu, "gamma": gamma }
        case "bennett":
            (rho, alpha, beta, lam) = optimizeBennett(L_tnd, L, 1, 1, n1, n2, 1, pi)
            bound = bennett(L_tnd, L, alpha, beta, n1, n2, lam, pi, rho)

            extra = { "alpha": alpha, "beta": beta, "lambda": lam }
        case _:
            raise f"Unknown bound type: {bound}"

    return rho, bound, extra

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

    res = np.array([[
        eval_bound(bound, dataset)
        for dataset in ds.keys()
    ] for bound in bounds])

    mv_bounds = arr2d_to_df(res[:,:,0], bounds, ds.keys())
    mv_risk   = arr2d_to_df(res[:,:,1], bounds, ds.keys())

    return mv_bounds, mv_risk
