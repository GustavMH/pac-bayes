#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from models.util import tandem_risks, gibbs_risks, oob_tandem_risks, oob_gibbs_risks
import bounds
from itertools import product
from tqdm import tqdm

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def calc_stats(val_preds, val_labels, test_preds, test_labels, n_iter_region=1000):
    n_models, n_examples = val_preds.shape
    n_cats = 1 + val_labels.max()
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
        "uni_bound": loss(uni, np.eye(n_cats)[val_preds], val_labels)
        + np.sqrt(np.log(2 / 0.05) / (2 * len(val_preds[0]))),
    }


def loss(rho, X, y):
    votes = (rho[:, None, None] * X).sum(0).argmax(-1)
    return 1 - (votes == y).mean()


# a = dict(np.load(Path("~/Downloads/pac-bayes-predictions/multimodel_eurosat.npz").expanduser()))
# (3 models, 10 runs, 2 training sets, 2 eval sets, 10 epochs, 1350 examples, 10 categories)
# res = a

# collect = []
# for model_n, train_n, eval_n, epoch_n in product([0, 1, 2], [0, 1], [0, 1], range(10)):
#     val_preds = res["validation"][model_n, :, train_n, eval_n, epoch_n].argmax(-1)
#     val_labels = res["val_labels"][eval_n]
#     test_preds = res["test"][model_n, :, train_n, eval_n, epoch_n].argmax(-1)
#     test_labels = res["test_labels"][eval_n]
#     collect.append(calc_stats(val_preds, val_labels, test_preds, test_labels))
# for train_n, eval_n, epoch_n in product([0, 1], [0, 1], range(10)):
#     val_preds = res["validation"][:, :, train_n, eval_n, epoch_n].reshape(30,1350,10).argmax(-1)
#     val_labels = res["val_labels"][eval_n]
#     test_preds = res["test"][:, :, train_n, eval_n, epoch_n].reshape(30,2700,10).argmax(-1)
#     test_labels = res["test_labels"][eval_n]
#     collect.append(calc_stats(val_preds, val_labels, test_preds, test_labels))

fo_loss = np.array([x["fo_test_loss"] for x in collect]).reshape(4, 2, 2, 10)
uni_loss = np.array([x["uni_test_loss"] for x in collect]).reshape(4, 2, 2, 10)
tnd_loss = np.array([x["tnd_test_loss"] for x in collect]).reshape(4, 2, 2, 10)
# tnd_rho = np.array([x["tnd_rho"] for x in collect]).reshape(4, 2, 2, 10, 10)
# fo_rho = np.array([x["fo_rho"] for x in collect]).reshape(4, 2, 2, 10, 10)

fig, axss = plt.subplots(2,4,sharex=True,sharey=True,figsize=(15.1,18),layout="compressed")

for ax, fo_L, uni_L, tnd_L in zip(axss.T.flat, fo_loss.reshape(8,2,10), uni_loss.reshape(8,2,10), tnd_loss.reshape(8,2,10)):
    ax.plot(fo_L[0], label="First-order")
    ax.plot(uni_L[0], label="Uniform")
    ax.plot(tnd_L[0], label="Tandem")
    #ax.set_ylim(.0,.20)

axss[0,0].set_title("MLP")
axss[0,1].set_title("ResNet18")
axss[0,2].set_title("ResNet18 Pretrained")
axss[0,3].set_title("All")
axss[0,0].legend()
plt.savefig("fig/test.png")
plt.close()

# fig, axss = plt.subplots(2,3,sharex=True,sharey=True,figsize=(5.1,3),layout="compressed")

# for ax, tnd_L, fo_L in zip(axss.T.flat, tnd_rho.reshape(6,2,10,10), fo_rho.reshape(6,2,10,10)):
#     ax.imshow(fo_L[0].T, label="First-order")
#     ax.imshow(tnd_L[0].T, label="Tandem")

# plt.savefig("fig/test.png")
# plt.close()
