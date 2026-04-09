#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from models.util import tandem_risks, gibbs_risks, split_bootstrap
from bounds import optimize_rho

# fit a linear plane, by labelling points in a with -1, b with +1, the seperator
# plane is when the regression gives 0.


def fit_plane(a,b):
    # Ordinary Least Squares
    n_a, n_b = a.shape[0], b.shape[0]
    y = np.block([-np.ones(n_a), np.ones(n_b)])
    X = np.block([[np.ones((n_a,1)),a],
                  [np.ones((n_b,1)),b]])
    return np.linalg.inv(X.T@X) @ X.T @ y

def fit_ensemble(a,b,n_members,rng):
    # bootstrap, n times
    a_s = rng.choice(a, size=(n_members, len(a)))
    b_s = rng.choice(b, size=(n_members, len(a)))

    # fit on subsets
    planes = [fit_plane(a,b) for a,b in zip(a_s, b_s)]

    # An ensemble is a matrix of b1, cx1, cy1, c...
    return np.array(planes)

def test_ensemble(ens, a, b):
    n_a, n_b = a.shape[0], b.shape[0]
    y = (np.block([-np.ones(n_a), np.ones(n_b)]) > 0)
    X = np.block([[np.ones((n_a,1)),a],
                  [np.ones((n_b,1)),b]])
    y_pred = (X @ ens.T).T > 0

    L, n1 = gibbs_risks(y_pred, y)
    L_tnd, n2 = tandem_risks(y_pred, y)

    return {
        "gibbs_risks": L,
        "n1": n1,
        "tandem_risks": L_tnd,
        "n2": n2,
        "test_predictions": y_pred,
        "test_labels": y
    }

def gen_pair_normals(rng, n_draws=100, n_dims=100, shift_rads=0):
    shift_radians = 0.1
    # cos sin some bullshit
    a = rng.multivariate_normal([ 1,*[0]*(n_dims-1)], np.eye(n_dims), size=n_draws)
    b = rng.multivariate_normal([-1,*[0]*(n_dims-1)], np.eye(n_dims), size=n_draws)
    return a, b



rng = np.random.default_rng()
a, b = gen_pair_normals(rng, 100, 10)

# train on a/b, ensemble in a_val/b_val, ensemble on a_shift/b_shift
a_val, b_val = gen_pair_normals(rng, 150, 10)
a_sft, b_sft = gen_pair_normals(rng, 150, shift_rads=0.1)

ens = fit_ensemble(a, b, 10, rng)
p1 = test_ensemble(ens, a, b)
p2 = test_ensemble(ens, a_val, b_val)
print(optimize_rho("tnd", p2)[1], p2["gibbs_risks"].round(2))

def plot_example_seperator(path="normal_shift.png"):
    rng = np.random.default_rng()

    a, b = gen_pair_normals(rng, 80)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    fig, axs = plt.subplots(1,2,figsize=(6,3.5))

    axs[0].scatter(*a.T[:2], alpha=0.9, color="black", marker="x", s=50)
    axs[0].scatter(*b.T[:2], alpha=0.9, color="black", marker="s", s=50)
    axs[1].scatter(*a.T[2:4], alpha=0.9, color="black", marker="x", s=50)
    axs[1].scatter(*b.T[2:4], alpha=0.9, color="black", marker="s", s=50)
    axs[0].plot([0,0],[-3,3], color="black", label="True seperator")

    for i in range(4,100,10):
        bias, cx, cy, c1, c2, *_ = fit_plane(a[:,:i],b[:,:i])
        line = lambda y: (-bias-cy*y)/cx
        axs[0].plot([line(-3),line(3)], [-3,3], color=(i/100, i/200, 1-i/100))
        line = lambda y: (-bias-c1*y)/c2
        axs[1].plot([line(-3),line(3)], [-3,3], color=(i/100, i/200, 1-i/100))

    axs[0].set_ylim([-3,3])
    axs[0].set_xlim([-3,3])
    axs[1].set_ylim([-3,3])
    axs[1].set_xlim([-3,3])
    #plt.legend()
    plt.savefig(path)
    plt.close()
