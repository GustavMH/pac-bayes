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

def fit_plane(a,b):
    # Ordinary Least Squares
    n_a, n_b = a.shape[0], b.shape[0]
    y = np.block([-np.ones(n_a), np.ones(n_b)])
    X = np.block([[np.ones((n_a,1)),a],
                  [np.ones((n_b,1)),b]])
    return np.linalg.inv(X.T@X) @ X.T @ y

def fit_ols_ensemble(a,b,n_members,rng):
    # bootstrap, n times
    a_s = rng.choice(a, size=(n_members, len(a)))
    b_s = rng.choice(b, size=(n_members, len(a)))

    # fit on subsets
    planes = [fit_plane(a,b) for a, b in zip(a_s, b_s)]

    # An ensemble is a matrix of b1, cx1, cy1, c...
    return np.array(planes)

def test_ols_ensemble(ens, a, b):
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

def plot_example_plugin():
    fig, axs = plt.subplots(1,2,figsize=(5.1,2.2),layout="tight",sharey=True)

    rng = np.random.default_rng()
    a, b = gen_pair_normals(rng, 150)
    val = gen_pair_normals(rng, 200)

    ests = fit_ensemble(a, b, 5, rng)
    p1 = test_ensemble(ests, *val)
    rho1, bound1, _ = optimize_rho("lambda", p1)
    print(bound1)

    t1 = axs[0].plot(*a[:, :2].T, marker = ".", c="gray",  linestyle = "none", label = "$\\mathcal{N}([+1, 0, \\dots, 0], I)$")
    t2 = axs[0].plot(*b[:, :2].T, marker = ".", c="black", linestyle = "none", label = "$\\mathcal{N}([-1, 0, \\dots, 0], I)$")

    #t3 = ax.plot([0,0], [3,-3], c = "black")
    for est, r in sorted(zip(ests, rho1), key=lambda x: x[1]):
        slope = plugin_to_slope(est)
        t3 = axs[0].plot([3/slope, -3/slope], [3,-3], c = "tab:blue",  label = "Plug-in")

    ests = fit_ols_ensemble(a, b, 5, rng)
    p2 = test_ols_ensemble(ests, *val)
    rho2, bound2, _ = optimize_rho("lambda", p2)
    print(bound2)
    t1 = axs[1].plot(*a[:, :2].T, marker = ".", c="gray", linestyle = "none", label = "$\\mathcal{N}([+1, 0, \\dots, 0], I)$")
    t2 = axs[1].plot(*b[:, :2].T, marker = ".", c="black",linestyle = "none", label = "$\\mathcal{N}([-1, 0, \\dots, 0], I)$")

    for (bias, cx, cy, *_), r in sorted(zip(ests, rho2), key=lambda x: x[1]):
        line = lambda y: (-bias-cy*y)/cx
        t4 = axs[1].plot([line(3), line(-3)], [3,-3], c = "tab:orange", label = "OLS")

    axs[0].set_ylim([-3,3])
    axs[0].set_xlim([-3,3])
    axs[1].set_ylim([-3,3])
    axs[1].set_xlim([-3,3])

    fig.legend(handles=t1+t2+t3+t4, ncol=2, loc="upper center", framealpha=1)

    plt.savefig("fig/example_plugin_est.pdf")
    plt.close()

fig, axs = plt.subplots(1,2,figsize=(5.1,2.2),layout="tight")

rng = np.random.default_rng()
a, b = gen_pair_normals(rng, 150)
val = gen_pair_normals(rng, 200)

ests = fit_ensemble(a, b, 5, rng)
p1 = test_ensemble(ests, *val)

l = np.zeros(400)
r = np.zeros(400)

im = np.zeros((400,400))
#t3 = ax.plot([0,0], [3,-3], c = "black")
for est in ests:
    slope = plugin_to_slope(est)
    l = np.minimum(l, np.linspace(-3,3,400)/slope)
    r = np.maximum(r, np.linspace(-3,3,400)/slope)
    im += np.add.outer(np.linspace(-3,3,400), np.linspace(-.5,.5,400)*-abs(slope)) < 0
    #t3 = axs[0].plot([3/slope, -3/slope], [3,-3], c = "tab:blue",  label = "Plug-in")

#axs[0].fill_betweenx(np.linspace(-3,3,400), r, l, color="grey", alpha=0.4)

axs[0].imshow(im/len(ests), interpolation="bicubic")
axs[0].contour(
    im/len(ests),
    levels=[0.01,0.5,0.99],
    colors=["black", "white", "black"],
    linestyles=["solid", "dotted", "solid"],
    linewidths=[.5,1,.5]
)
axs[0].set_title("Plug-in")

ests = fit_ols_ensemble(a, b, 5, rng)
p2 = test_ols_ensemble(ests, *val)
l = np.zeros(400)
r = np.zeros(400)


im = np.zeros((400,400))

for (bias, cx, cy, *_) in ests:
    line = lambda y: (-bias-cy*y)/cx
    l = np.minimum(l, line(np.linspace(-3,3,400)))
    r = np.maximum(r, line(np.linspace(-3,3,400)))
    im += (np.add.outer(np.linspace(-3,3,400)*cy, np.linspace(-.5,.5,400)*cx)+bias) < 0
    #t4 = axs[1].plot([line(3), line(-3)], [3,-3], c = "tab:orange", label = "OLS")

axs[1].imshow(im/len(ests), interpolation="bicubic")
axs[1].contour(
    im/len(ests),
    levels=[0.01,0.5,0.99],
    colors=["black", "white", "black"],
    linestyles=["solid", "dotted", "solid"],
    linewidths=[.5,1,.5]
)
axs[1].set_title("OLS")
#axs[1].fill_betweenx(np.linspace(-3,3,200), r, l, color="grey", alpha=0.4)

plt.savefig("fig/plugin_region.pdf")
plt.close()
