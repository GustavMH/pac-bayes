#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def gen_pair_normals(rng, n_draws=200, n_dims=100, shift_rads=0):
    x, y = np.cos(shift_rads), np.sin(shift_rads)
    a = rng.multivariate_normal([+x,+y,*[0]*(n_dims-2)], np.eye(n_dims), size=n_draws)
    b = rng.multivariate_normal([-x,-y,*[0]*(n_dims-2)], np.eye(n_dims), size=n_draws)
    return a, b

def fit_plugin_estimator(pos, neg):
    return pos.mean(0) - neg.mean(0)

def plugin_to_slope(mean):
    return -mean[0]/mean[1]

def fit_ensemble(a,b,n_members,rng):
    # bootstrap, n times
    a_s = rng.choice(a, size=(n_members, len(a)))
    b_s = rng.choice(b, size=(n_members, len(a)))

    # fit on subsets
    planes = [fit_plugin_estimator(a,b) for a,b in zip(a_s, b_s)]

    # An ensemble is a matrix of b1, cx1, cy1, c...
    return np.array(planes)

def test_ensemble(ens, a, b):
    n_a, n_b = a.shape[0], b.shape[0]
    y = (np.block([np.ones(n_a), -np.ones(n_b)]) > 0)
    X = np.block([[a], [b]])
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

def whatever():
    rng = np.random.default_rng()
    a, b = gen_pair_normals(rng)

    ests = fit_ensemble(a, b, 10, rng)
    p1 = test_ensemble(ests, *gen_pair_normals(rng))
    rho1, bound, _ = optimize_rho("lambda", p1)
    print(rho1)

    fig, ax = plt.subplots(1,1,figsize=(5.1,3.2),layout="tight", dpi=200)

    t1 = ax.plot(*a[:, :2].T, marker = ".", linestyle = "none", label = "$\\mathcal{N}([+1, 0, \\dots, 0], I)$")
    t2 = ax.plot(*b[:, :2].T, marker = ".", linestyle = "none", label = "$\\mathcal{N}([-1, 0, \\dots, 0], I)$")

    #t3 = ax.plot([0,0], [3,-3], c = "black")
    for est, r in zip(ests, rho1):
        slope = plugin_to_slope(est)
        t3 = ax.plot([3/slope, -3/slope], [3,-3], c = "tab:green", alpha=r/rho1.max())

    ax.set_ylim([-3,3])
    ax.set_xlim([-1,1])

    plt.savefig("fig/example_plugin_est.pdf")
    plt.close()

def plot_example_plugin_est():
    rng = np.random.default_rng()
    a, b = gen_pair_normals(rng)

    est = fit_plugin_estimator(a,b)
    slope = plugin_to_slope(est)

    fig, ax = plt.subplots(1,1,figsize=(5.1,3.2),layout="tight", dpi=200)

    t1 = ax.plot(*a[:, :2].T, marker = ".", linestyle = "none", label = "$\\mathcal{N}([+1, 0, \\dots, 0], I)$")
    t2 = ax.plot(*b[:, :2].T, marker = ".", linestyle = "none", label = "$\\mathcal{N}([-1, 0, \\dots, 0], I)$")
    t3 = ax.plot([4/slope, -4/slope], [4,-4])

    ax.set_ylim([-4,4])
    ax.set_xlim([-4,4])


fig, axs = plt.subplots(1,4,figsize=(5.1,1.5), layout="tight", sharex=True, sharey=True, dpi=200)

rng = np.random.default_rng()
for ax in axs:
    a, b = gen_pair_normals(rng, 20)
    t1 = ax.plot(*a[:, :2].T, marker = ".", linestyle = "none", label = "$\\mathcal{N}([+1, 0, \\dots, 0], I)$")
    t2 = ax.plot(*b[:, :2].T, marker = ".", linestyle = "none", label = "$\\mathcal{N}([-1, 0, \\dots, 0], I)$")
    t3 = ax.plot()
    est = fit_plugin_estimator(a,b)
    slope = plugin_to_slope(est)
    t3 = ax.plot([3/slope, -3/slope], [3,-3])

ax.set_ylim([-3,3])
ax.set_xlim([-3,3])

plt.savefig("fig/example_plugin_est.pdf")
plt.close()
