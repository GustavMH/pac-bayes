#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models.util import tandem_risks, gibbs_risks
import bounds

try:
    res
except NameError:
    path = Path("~/Downloads/eurosat_chroma_shift.npz").expanduser()
    res = dict(np.load(path))

# (10 runs, 5 training mixes, 2 test sets, 15 snapshots, 2700 examples, 10 categories)
# Take model trained on A, see performance on A, trained for 10 epochs

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
    opt = np.array([loss(rho, X, y) for rho in simplex])
    return np.max(opt), np.min(opt)

fig, axss = plt.subplots(2,1,figsize=(5.5,3.5),sharex=True,sharey=True)

labels_A, labels_B = res["val_labels"]
tst_lab_A, tst_lab_B = res["test_labels"]

vmin = 1
vmax = 0
collect_risks_A = np.zeros((30,10))
collect_risks_B = np.zeros((30,10))
collect_bounds_A = np.zeros(30)
collect_bounds_B = np.zeros(30)
collect_rhos_A = np.zeros((30,10))
collect_rhos_B = np.zeros((30,10))
collect_reg_A = np.zeros((30,2))
collect_reg_B = np.zeros((30,2))
collect_res_A = np.zeros(30)
collect_res_B = np.zeros(30)
collect_uni_A = np.zeros(30)
collect_uni_B = np.zeros(30)
for i in range(30):
    preds_A = res["validation"][:,0,0,i].argmax(-1)
    preds_B = res["validation"][:,0,1,i].argmax(-1)
    test_A = res["test"][:,0,0,i].argmax(-1)
    test_B = res["test"][:,0,1,i].argmax(-1)

    risks_A, n1_A = gibbs_risks(preds_A, labels_A)
    tnd_A, n2_A = tandem_risks(preds_A, labels_A)
    collect_risks_A[i] = risks_A
    params_A = {"tandem_risks": tnd_A, "n2": n2_A, "gibbs_risks": risks_A, "n1": n1_A}
    rho_A, bound_A, _ = bounds.optimize_rho("tnd", params_A)
    collect_bounds_A[i] = bound_A
    collect_rhos_A[i] = rho_A
    collect_reg_A[i] = est_feasible_region(np.eye(10)[test_A], tst_lab_A)
    collect_res_A[i] = loss(rho_A, np.eye(10)[test_A], tst_lab_A)
    collect_uni_A[i] = loss(np.ones(10) / 10, np.eye(10)[test_A], tst_lab_A)

    risks_B, n1_B = gibbs_risks(preds_B, labels_B)
    tnd_B, n2_B = tandem_risks(preds_B, labels_B)
    collect_risks_B[i] = risks_B
    params_B = {"tandem_risks": tnd_B, "n2": n2_B, "gibbs_risks": risks_B, "n1": n1_B}
    rho_B, bound_B, _ = bounds.optimize_rho("tnd", params_B)
    collect_bounds_B[i] = bound_B
    collect_rhos_B[i] = rho_B
    collect_reg_B[i] = est_feasible_region(np.eye(10)[test_B], tst_lab_B)
    collect_res_B[i] = loss(rho_B, np.eye(10)[test_B], tst_lab_B)
    collect_uni_B[i] = loss(np.ones(10) / 10, np.eye(10)[test_B], tst_lab_B)

    print(f"{bound_A=} {bound_B=}")

idx = np.argsort(collect_risks_B[-1])
vmin = np.min([collect_risks_A, collect_risks_B])
vmax = np.max([collect_risks_A, collect_risks_B])

fig.supxlabel("Epoch")
fig.supylabel("Training run")
axss[0].imshow(collect_risks_A.T[idx], interpolation='nearest', aspect='auto')
m = axss[1].imshow(collect_risks_B.T[idx], interpolation='nearest', aspect='auto')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(m, cax=cbar_ax)

plt.savefig("eurosat_IN1K.png")
plt.close()


fig, ax = plt.subplots(1,1,figsize=(5.5,4),sharex=True,sharey=True)
plt.title("Voting ensembles, 10 members, before shift")
ax.plot(collect_res_A, label="Weighted loss A")
ax.plot(collect_uni_A, label="Uniform loss A")
ax.fill_between(np.arange(30), collect_reg_A[:,0], collect_reg_A[:,1], label="Est. feasible region A", alpha=0.2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.legend()

plt.savefig("fig/eurosat_perf_A.png")
plt.close()

fig, ax = plt.subplots(1,1,figsize=(5.5,4),sharex=True,sharey=True)
plt.title("Voting ensembles, 10 members, after shift")
ax.plot(collect_res_B, label="Weighted loss B")
ax.plot(collect_uni_B, label="Uniform loss B")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

ax.fill_between(np.arange(30), collect_reg_B[:,0], collect_reg_B[:,1], label="Est. feasible region B", alpha=0.2)
ax.legend()
plt.savefig("fig/eurosat_perf_B.png")
plt.close()

fig, ax = plt.subplots(1,1,figsize=(5.5,4),sharex=True,sharey=True,layout="tight")
plt.title("Voting ensembles, optimized 'tnd' bound, 10 members")
ax.plot(collect_res_A, label="Weighted loss A")
ax.fill_between(np.arange(30), collect_res_A, collect_bounds_A, label="Loss bound A", alpha=0.2)

ax.plot(collect_res_B, label="Weighted loss B")
ax.fill_between(np.arange(30), collect_res_B, collect_bounds_B, label="Loss bound B", alpha=0.2)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
from matplotlib.ticker import PercentFormatter
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.legend()
plt.savefig("fig/eurosat_bounds_B.png")
plt.close()


def plot_voting():

    X=softmax(res["test"][:,0,0,0])
    y=res["test_labels"][0]

    all_idx = ~(Xm == Xm[0]).all(0)
    Xs, ys = X[:, all_idx], y[all_idx]

    loss(np.ones(10)/10, Xs, ys)

    # Søg efter optimal vægt vha. binær søgning?

    # res = minimize(lambda rho: np.abs(np.matvec((X-y).T, softmax(rho)).sum(-1)).sum(), np.ones(10) / 10)
    ind = [i for i, _ in sorted(enumerate(np.concat([ys[None,:],Xs.argmax(-1)]).T), key=lambda x: tuple(x[1]))]
    ind = np.array(ind)

    fig, ax = plt.subplots(1,1,figsize=(5.5,2),sharex=True,sharey=True,layout="tight")
    ax.imshow(np.take_along_axis(Xs.argmax(-1).T, ind[:,None], axis=0).T, cmap="tab10", interpolation="nearest", aspect="auto")
    plt.savefig("fig/votes.png")
    plt.close()
