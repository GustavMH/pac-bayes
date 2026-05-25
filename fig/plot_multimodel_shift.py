#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from models.util import tandem_risks, gibbs_risks, oob_tandem_risks, oob_gibbs_risks
import bounds
from itertools import product
from tqdm import tqdm
from scipy.stats import mannwhitneyu as utest

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


#res = dict(np.load(Path("~/Downloads/pac-bayes-predictions/multimodel_eurosat_2.npz").expanduser()))
#(3 models, 10 runs, 2 training sets, 2 eval sets, 10 epochs, 1350 examples, 10 categories)

def load_multimodel(res):
    collect = []
    for model_n, train_n, eval_n, epoch_n in product(range(6), [0, 1], [0, 1], range(10)):
        val_preds = res["validation"][model_n, :, train_n, eval_n, epoch_n].argmax(-1)
        val_labels = res["val_labels"][eval_n]
        test_preds = res["test"][model_n, :, train_n, eval_n, epoch_n].argmax(-1)
        test_labels = res["test_labels"][eval_n]
        collect.append(calc_stats(val_preds, val_labels, test_preds, test_labels))
    for train_n, eval_n, epoch_n in product([0, 1], [0, 1], range(10)):
        val_preds = res["validation"][:, :, train_n, eval_n, epoch_n].reshape(60,1350,10).argmax(-1)
        val_labels = res["val_labels"][eval_n]
        test_preds = res["test"][:, :, train_n, eval_n, epoch_n].reshape(60,2700,10).argmax(-1)
        test_labels = res["test_labels"][eval_n]
        collect.append(calc_stats(val_preds, val_labels, test_preds, test_labels))
    return collect

def load_multimodel_snapshots(res):
    collect = []
    for model_n, train_n, eval_n, run_n in product(range(6), [0, 1], [0, 1], range(10)):
        val_preds = res["validation"][model_n, run_n, train_n, eval_n, :].argmax(-1)
        val_labels = res["val_labels"][eval_n]
        test_preds = res["test"][model_n, run_n, train_n, eval_n, :].argmax(-1)
        test_labels = res["test_labels"][eval_n]
        collect.append(calc_stats(val_preds, val_labels, test_preds, test_labels))
    for train_n, eval_n, run_n in product([0, 1], [0, 1], range(10)):
        val_preds = res["validation"][:, run_n, train_n, eval_n, :].reshape(60,-1,10).argmax(-1)
        val_labels = res["val_labels"][eval_n]
        test_preds = res["test"][:, run_n, train_n, eval_n, :].reshape(60,-1,10).argmax(-1)
        test_labels = res["test_labels"][eval_n]
        collect.append(calc_stats(val_preds, val_labels, test_preds, test_labels))
    return collect

def load_mult(res, idx="eurosat"):
    """CIFAR+IMDB last uniform, last tandem, all tandem, all uniform"""

    model_idx = 1 if idx == "eurosat" else 2

    best = []
    for iter_n, n_models in product(range(10), range(2,11)):
        for eval_n, run_n in product([0, 1], range(10)):
            idx = np.random.permutation(10)[:n_models]

            val_preds = res["validation"][model_idx, run_n, 0, eval_n, idx].argmax(-1)
            val_labels = res["val_labels"][eval_n]
            test_preds = res["test"][model_idx, run_n, 0, eval_n, idx].argmax(-1)
            test_labels = res["test_labels"][eval_n]
            best.append(calc_stats(val_preds, val_labels, test_preds, test_labels))

    collect = []
    for iter_n, n_models in product(range(10), range(2,31)):
        for eval_n, run_n in product([0, 1], range(10)):
            idx = np.random.permutation(60)[:n_models]
            idx_a = idx % 10
            idx_b = idx // 10

            val_preds = res["validation"][idx_b, run_n, 0, eval_n, idx_a].argmax(-1)
            val_labels = res["val_labels"][eval_n]
            test_preds = res["test"][idx_b, run_n, 0, eval_n, idx_a].argmax(-1)
            test_labels = res["test_labels"][eval_n]

            collect.append(calc_stats(val_preds, val_labels, test_preds, test_labels))

    return np.array(best).reshape(10,9,2,10), np.array(collect).reshape(10,29,2,10)

def load_worse():
    pass

def simple_ensembles():
    #collect = load_multimodel()

    fo_loss = np.array([x["fo_test_loss"] for x in collect]).reshape(7, 2, 2, 10)
    uni_loss = np.array([x["uni_test_loss"] for x in collect]).reshape(7, 2, 2, 10)
    tnd_loss = np.array([x["tnd_test_loss"] for x in collect]).reshape(7, 2, 2, 10)
    # tnd_rho = np.array([x["tnd_rho"] for x in collect]).reshape(4, 2, 2, 10, 10)
    # fo_rho = np.array([x["fo_rho"] for x in collect]).reshape(4, 2, 2, 10, 10)

    fig, axs = plt.subplots(1,7,sharex=True,sharey="row",figsize=(5.5,2),layout="compressed")

    titles = ["Conv.", "Res.", "Dense.", "Efficient.", "ConvNeXt", "Shuffle.", "ALL"]

    for ax, fo_L, uni_L, tnd_L, title in zip(axs, fo_loss, uni_loss, tnd_loss, titles):
        ax.plot(fo_L[0,0], label="First-order",c="tab:blue")
        ax.plot(uni_L[0,0], label="Uniform",c="tab:orange")
        ax.plot(tnd_L[0,0], label="Tandem",c="tab:green")
        ax.plot(fo_L[0,1], label="First-order",c="tab:blue", linestyle=(0, (1,1)))
        ax.plot(uni_L[0,1], label="Uniform",c="tab:orange", linestyle=(0, (1,1)))
        ax.plot(tnd_L[0,1], label="Tandem",c="tab:green", linestyle=(0, (1,1)))
        ax.set_xticks([0,4,9],[1,5,10])
        ax.get_xticklabels()[-1].set_horizontalalignment("right")
        print(f"{title} tnd vs. fo  {utest(tnd_L[0,0], fo_L[0,0]).pvalue:.2f} {(tnd_L[0,0]-fo_L[0,0]).mean():.2f}")
        print(f"{title} tnd vs. uni {utest(tnd_L[0,0], uni_L[0,0]).pvalue:.2f} {(tnd_L[0,0]-uni_L[0,0]).mean():.2f}")
        print(f"{title} tnd vs. fo  {utest(tnd_L[0,1], fo_L[0,1]).pvalue:.2f} {(tnd_L[0,1]-fo_L[0,1]).mean():.2f}")
        print(f"{title} tnd vs. uni {utest(tnd_L[0,1], uni_L[0,1]).pvalue:.2f} {(tnd_L[0,1]-uni_L[0,1]).mean():.2f}")
        ax.set_title(title)

    from matplotlib.lines import Line2D
    legend = axs[2].legend(
        framealpha=1,
        loc="upper right",
        handles=[
            Line2D([],[],c="tab:blue"),
            Line2D([],[],c="tab:orange"),
            Line2D([],[],c="tab:green"),
        ],
        labels=[
            "First-order",
            "Uniform",
            "Tandem",
        ]
    )
    legend.set_in_layout(False)
    legend = axs[-1].legend(
        framealpha=1,
        loc="upper right",
        handles=[
            Line2D([],[],c="black"),
            Line2D([],[],c="black", linestyle=(0,(1,1))),
        ],
        labels=[
            "EuroSat A",
            "EuroSat B"
        ]
    )
    legend.set_in_layout(False)

    plt.savefig("fig/shift_simple_ens.pdf")
    plt.close()

    # fig, axss = plt.subplots(2,3,sharex=True,sharey=True,figsize=(5.1,3),layout="compressed")

    # for ax, tnd_L, fo_L in zip(axss.T.flat, tnd_rho.reshape(6,2,10,10), fo_rho.reshape(6,2,10,10)):
    #     ax.imshow(fo_L[0].T, label="First-order")
    #     ax.imshow(tnd_L[0].T, label="Tandem")

    # plt.savefig("fig/test.png")
    # plt.close()

def smear_plot_sse():
    fo_loss = np.array([x["fo_test_loss"] for x in collect]).reshape(7, 2, 2, 10)
    uni_loss = np.array([x["uni_test_loss"] for x in collect]).reshape(7, 2, 2, 10)
    tnd_loss = np.array([x["tnd_test_loss"] for x in collect]).reshape(7, 2, 2, 10)
    # tnd_rho = np.array([x["tnd_rho"] for x in collect]).reshape(4, 2, 2, 10, 10)
    # fo_rho = np.array([x["fo_rho"] for x in collect]).reshape(4, 2, 2, 10, 10)

    fig, ax = plt.subplots(1,1,sharex=True,sharey="row",figsize=(5.5,2),layout="compressed")

    titles = ["Conv.", "Res.", "Dense.", "Efficient.", "ConvNeXt", "Shuffle.", "ALL"]

    scatter = np.random.uniform(0,0.2,size=10)*0
    for fo_L, uni_L, tnd_L, title, i in zip(fo_loss, uni_loss, tnd_loss, titles, range(7)):
        ax.plot(scatter+i-0.2, fo_L[0,0], c="tab:blue", linestyle="none", marker="o", alpha=0.5)
        ax.plot(scatter+i-0.0, uni_L[0,0], c="tab:orange", linestyle="none", marker="o", alpha=0.5)
        ax.plot(scatter+i+0.2, tnd_L[0,0], c="tab:green", linestyle="none", marker="o", alpha=0.5)
        ax.plot(scatter+i-0.2, fo_L[0,1], c="tab:blue", linestyle="none", marker="o", alpha=0.5)
        ax.plot(scatter+i-0.0, uni_L[0,1], c="tab:orange", linestyle="none", marker="o", alpha=0.5)
        ax.plot(scatter+i+0.2, tnd_L[0,1], c="tab:green", linestyle="none", marker="o", alpha=0.5)
        ax.get_xticklabels()[-1].set_horizontalalignment("right")
        ax.set_ylim(bottom=0, top=0.5)
        print(f"{title} tnd vs. fo  {utest(tnd_L[0,0], fo_L[0,0]).pvalue:.2f} {(tnd_L[0,0]-fo_L[0,0]).mean():.2f}")
        print(f"{title} tnd vs. uni {utest(tnd_L[0,0], uni_L[0,0]).pvalue:.2f} {(tnd_L[0,0]-uni_L[0,0]).mean():.2f}")
        print(f"{title} tnd vs. fo  {utest(tnd_L[0,1], fo_L[0,1]).pvalue:.2f} {(tnd_L[0,1]-fo_L[0,1]).mean():.2f}")
        print(f"{title} tnd vs. uni {utest(tnd_L[0,1], uni_L[0,1]).pvalue:.2f} {(tnd_L[0,1]-uni_L[0,1]).mean():.2f}")
        ax.set_title(title)

    plt.savefig("fig/shift_snapshot_ens.pdf")
    plt.close()

def shift_diff_plot(collect):
    fo_loss = np.array([x["fo_test_loss"] for x in collect]).reshape(7, 2, 2, 10)
    uni_loss = np.array([x["uni_test_loss"] for x in collect]).reshape(7, 2, 2, 10)
    tnd_loss = np.array([x["tnd_test_loss"] for x in collect]).reshape(7, 2, 2, 10)

    idx = np.concat((np.argsort(uni_loss[:6,0,1,:].mean(1)),[6]))

    titles = np.array(["Conv.", "Res.", "Dense.", "Efficient.", "ConvNeXt", "Shuffle.", "ALL"])

    fo_loss = fo_loss[idx]
    uni_loss = uni_loss[idx]
    tnd_loss = tnd_loss[idx]
    titles = titles[idx]

    # best_loss = np.array([x["best_test_loss"] for x in collect]).reshape(7, 2, 2, 10)
    # tnd_rho = np.array([x["tnd_rho"] for x in collect]).reshape(4, 2, 2, 10, 10)
    # fo_rho = np.array([x["fo_rho"] for x in collect]).reshape(4, 2, 2, 10, 10)

    fig, axs = plt.subplots(2,1,sharex=True,sharey="row",figsize=(5.5,4),layout="compressed")


    scatter = np.random.uniform(0,0.2,size=10)*0
    for fo_L, uni_L, tnd_L, best_L, title, i in zip(fo_loss, uni_loss, tnd_loss, tnd_loss, titles, range(7)):
        axs[0].plot(scatter+i-0.2, fo_L[0,0], c="tab:blue", linestyle="none", marker="o", alpha=0.5)
        axs[0].plot(scatter+i-0.0, uni_L[0,0], c="tab:orange", linestyle="none", marker="o", alpha=0.5)
        axs[0].plot(scatter+i+0.2, tnd_L[0,0], c="tab:green", linestyle="none", marker="o", alpha=0.5)
        axs[0].plot(scatter+i-0.2, fo_L[0,1], c="tab:blue", linestyle="none", marker="s", alpha=0.5)
        axs[0].plot(scatter+i+0.0, uni_L[0,1], c="tab:orange", linestyle="none", marker="s", alpha=0.5)
        axs[0].plot(scatter+i+0.2, tnd_L[0,1], c="tab:green", linestyle="none", marker="s", alpha=0.5)

    # for fo_L, uni_L, tnd_L, best_L, title, i in zip(fo_loss, uni_loss, tnd_loss, tnd_loss, titles, range(7)):
    #     axs[1].plot(scatter+i-0.3, -uni_L[0,0]+fo_L[0,0], c="tab:blue", linestyle="none", marker="o", alpha=0.5)
    #     axs[1].plot(scatter+i-0.1, -uni_L[0,0]+tnd_L[0,0], c="tab:green", linestyle="none", marker="o", alpha=0.5)
    #     #ax.plot(scatter+i-0.0, uni_L[0,0]-best_L[0,0], c="tab:red", linestyle="none", marker="o", alpha=0.5)
    #     #ax.plot(scatter+i-0.0, uni_L[0,1]-best_L[0,1], c="tab:red", linestyle="none", marker="o", alpha=0.5)
    #     axs[1].plot(scatter+i+0.1, -uni_L[0,1]+fo_L[0,1], c="tab:blue", linestyle="none", marker="s", alpha=0.5)
    #     axs[1].plot(scatter+i+0.3, -uni_L[0,1]+tnd_L[0,1], c="tab:green", linestyle="none", marker="s", alpha=0.5)
    #     print(f"{title}       tnd vs. fo  {utest(tnd_L[0,0], fo_L[0,0]).pvalue:.2f} {(tnd_L[0,0]-fo_L[0,0]).mean()*100:.2f}")
    #     print(f"{title} SHIFT tnd vs. uni {utest(tnd_L[0,0], uni_L[0,0]).pvalue:.2f} {(tnd_L[0,0]-uni_L[0,0]).mean()*100:.2f}")
    #     print(f"{title}       tnd vs. fo  {utest(tnd_L[0,1], fo_L[0,1]).pvalue:.2f} {(tnd_L[0,1]-fo_L[0,1]).mean()*100:.2f}")
    #     print(f"{title} SHIFT tnd vs. uni {utest(tnd_L[0,1], uni_L[0,1]).pvalue:.2f} {(tnd_L[0,1]-uni_L[0,1]).mean()*100:.2f}")

    for fo_L, uni_L, tnd_L, best_L, title, i in zip(fo_loss, uni_loss, tnd_loss, tnd_loss, titles, range(7)):
        axs[1].plot(scatter+i-0.3, (fo_L[0,0]-uni_L[0,0])/(uni_L[0,1]-uni_L[0,0]), c="tab:blue", linestyle="none", marker="o", alpha=0.5)
        axs[1].plot(scatter+i-0.1, (tnd_L[0,0]-uni_L[0,0])/(uni_L[0,1]-uni_L[0,0]), c="tab:green", linestyle="none", marker="o", alpha=0.5)
        axs[1].plot(scatter+i+0.1, (fo_L[0,1]-uni_L[0,1])/(uni_L[0,1]-uni_L[0,0]), c="tab:blue", linestyle="none", marker="s", alpha=0.5)
        axs[1].plot(scatter+i+0.3, (tnd_L[0,1]-uni_L[0,1])/(uni_L[0,1]-uni_L[0,0]), c="tab:green", linestyle="none", marker="s", alpha=0.5)
        print(f"{((fo_L[0,1]-uni_L[0,1])/(uni_L[0,1]-uni_L[0,0])).mean():.4f}")
        print(f"{utest(fo_L[0,1],uni_L[0,1]).pvalue:.4f}")

    for ax in axs:
        ax.axvline(5.5, linestyle="dotted", c="gray", alpha=0.5)

    for tnd, title in zip(tnd_loss[:6], titles):
        print(f"{((tnd[0,1]-tnd_loss[6,0,1]).mean()*100).round(2)} pvalue {utest(tnd[0,1],tnd_loss[6,0,1]).pvalue:.4f}")

    axs[0].legend(
        ncols=2,
        handles=[
            Line2D([],[], c="tab:blue"),
            Line2D([],[], c="tab:orange"),
            Line2D([],[], c="tab:green"),
            Line2D([],[], c="black", linestyle="none", marker="o"),
            Line2D([],[], c="black", linestyle="none", marker="s"),
        ],
        labels=[
            "First-order",
            "Uniform",
            "Tandem",
            "Before shift",
            "After shift",
        ]
    )
    axs[0].set_ylabel("Loss")
    axs[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axs[0].set_title("Loss increase under distribution shift")
    # axs[1].plot([-0.5,6.5], [0,0], linestyle="dotted", c="gray", alpha=0.5)
    axs[1].set_xlim(-0.5,6.5)
    # axs[1].set_ylim(bottom=-0.062)
    axs[1].set_xticks(range(7), titles)
    # axs[1].set_title("Change compared with uniform weights")
    # axs[1].set_ylabel("Loss change")
    # axs[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol="pp"))
    axs[1].set_ylabel("Loss change / Shift size")
    axs[1].plot([-0.5,6.5], [0,0], linestyle="dotted", c="gray", alpha=0.5)
    axs[1].set_title("Normalized change compared with uniform weights")
    axs[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    #plt.savefig("fig/shift_snapshot_ens.pdf")
    plt.close()

def shift_diff_plot(collect):
    fo_loss = np.array([x["fo_test_loss"] for x in collect]).reshape(7, 2, 2, 10)
    uni_loss = np.array([x["uni_test_loss"] for x in collect]).reshape(7, 2, 2, 10)
    tnd_loss = np.array([x["tnd_test_loss"] for x in collect]).reshape(7, 2, 2, 10)

    idx = np.concat((np.argsort(uni_loss[:6,0,1,:].mean(1)),[6]))

    titles = np.array(["Conv.", "Res.", "Dense.", "Efficient.", "ConvNeXt", "Shuffle.", "ALL"])

    fo_loss = fo_loss[idx]
    uni_loss = uni_loss[idx]
    tnd_loss = tnd_loss[idx]
    titles = titles[idx]


def quad_shift_tnd():
    fig, axs = plt.subplots(2,2,figsize=(5.5,2.5),layout="constrained", sharex=True)

    for ax, mult, best_mult in zip(axs.T, [mult_eurosat, mult_imagenette], [best_mult_eurosat, best_mult_imagenette]):
        tnd_loss = np.array([x["tnd_test_loss"] for x in mult.flat]).reshape(10, 29, 2, 10)

        µ_tnd = tnd_loss.mean(0).mean(-1)
        std_tnd = tnd_loss.transpose(1,2,3,0).reshape(29,2,100).std(-1, ddof=1)

        best = np.array([a["tnd_test_loss"] for a in best_mult.flat]).reshape(10,9,2,10)
        µ_best = best.mean(0).mean(-1)
        std_best = best.transpose(1,2,3,0).reshape(9,2,100).std(-1, ddof=1)

        for i in range(29):
            p = utest(best[:,-1,1,:].flat, tnd_loss[:,i,1,:].flat).pvalue
            µ = (best[:,-1,1,:]-tnd_loss[:,i,1,:]).mean()
            print(f"{i:2d} {p:.4f} {µ:+.4f}")

        ax[0].plot(µ_best[:,1], c="tab:brown")
        ax[0].fill_between(np.arange(9), µ_best[:,1]+std_best[:,1], µ_best[:,1]-std_best[:,1], color="tab:brown", alpha=0.2)
        ax[0].fill_between([8,28], µ_best[-1,1]+std_best[-1,1], µ_best[-1,1]-std_best[-1,1], color="tab:brown", alpha=0.2)
        ax[0].plot([8,28], [µ_best[-1,1],µ_best[-1,1]], linestyle="dashed", c="tab:brown")

        ax[1].plot(µ_best[:,0], c="tab:brown")
        ax[1].fill_between(np.arange(9), µ_best[:,0]+std_best[:,0], µ_best[:,0]-std_best[:,0], color="tab:brown", alpha=0.2)
        ax[1].fill_between([8,28], µ_best[-1,0]+std_best[-1,0], µ_best[-1,0]-std_best[-1,0], color="tab:brown", alpha=0.2)
        ax[1].plot([8,28], [µ_best[-1,0],µ_best[-1,0]], linestyle="dashed", c="tab:brown")

        ax[0].plot(µ_tnd[:,1], label=j, c="tab:purple")
        ax[0].fill_between(np.arange(29), µ_tnd[:,1]+std_tnd[:,1], µ_tnd[:,1]-std_tnd[:,1], color="tab:purple", alpha=0.2)

        ax[1].plot(µ_tnd[:,0], label=j, c="tab:purple")
        ax[1].fill_between(np.arange(29), µ_tnd[:,0]+std_tnd[:,0], µ_tnd[:,0]-std_tnd[:,0], color="tab:purple", alpha=0.2)


    axs[0,0].set_ylim(top=0.36)
    axs[0,0].set_ylabel("λAfter shift")
    axs[1,0].set_ylabel("Before shift")
    axs[1,0].set_xlabel("No. of ensemble members")
    axs[1,1].set_xlabel("No. of ensemble members")
    for ax in axs.flat:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    axs[1,0].set_xticks([0,3,8,13,18,23,28],[2,5,10,15,20,25,30])
    axs[1,1].set_xticks([0,3,8,13,18,23,28],[2,5,10,15,20,25,30])
    axs[0,1].legend(
        handles=[
            Line2D([],[],c="tab:brown"),
            Line2D([],[],c="tab:purple"),
        ],
        labels = [
            "Best model type",
            "All models",
        ]
    )
    plt.savefig("fig/double.png", dpi=300)
    plt.close()


def double_shift():
    fig, axs = plt.subplots(1,2,figsize=(5.5,1.8),layout="constrained", sharex=True)

    for ax, mult, best_mult in zip(axs.T, [mult_eurosat, mult_imagenette], [best_mult_eurosat, best_mult_imagenette]):
        tnd_loss = np.array([x["tnd_test_loss"] for x in mult.flat]).reshape(10, 29, 2, 10)

        µ_tnd = tnd_loss.mean(0).mean(-1)
        std_tnd = tnd_loss.transpose(1,2,3,0).reshape(29,2,100).std(-1, ddof=1)

        best = np.array([a["tnd_test_loss"] for a in best_mult.flat]).reshape(10,9,2,10)
        µ_best = best.mean(0).mean(-1)
        std_best = best.transpose(1,2,3,0).reshape(9,2,100).std(-1, ddof=1)

        for i in range(29):
            p = utest(best[:,-1,1,:].flat, tnd_loss[:,i,1,:].flat).pvalue
            µ = (best[:,-1,1,:]-tnd_loss[:,i,1,:]).mean()
            print(f"{i:2d} {p:.4f} {µ:+.4f}")

        ax.plot(µ_best[:,1], c="tab:brown")
        ax.fill_between(np.arange(9), µ_best[:,1]+std_best[:,1], µ_best[:,1]-std_best[:,1], color="tab:brown", alpha=0.2)
        ax.fill_between([8,28], µ_best[-1,1]+std_best[-1,1], µ_best[-1,1]-std_best[-1,1], color="tab:brown", alpha=0.2)
        ax.plot([8,28], [µ_best[-1,1],µ_best[-1,1]], linestyle="dashed", c="tab:brown")

        ax.plot(µ_tnd[:,1], label=j, c="tab:purple")
        ax.fill_between(np.arange(29), µ_tnd[:,1]+std_tnd[:,1], µ_tnd[:,1]-std_tnd[:,1], color="tab:purple", alpha=0.2)


    axs[0].set_ylim(top=0.36)
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("No. of ensemble members")
    axs[1].set_xlabel("No. of ensemble members")
    axs[0].set_title("EuroSAT")
    axs[1].set_title("Imagenette")

    for ax in axs.flat:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    axs[0].set_xticks([0,3,8,13,18,23,28],[2,5,10,15,20,25,30])
    axs[1].set_xticks([0,3,8,13,18,23,28],[2,5,10,15,20,25,30])

    axs[1].legend(
        handles=[
            Line2D([],[],c="tab:brown"),
            Line2D([],[],c="tab:purple"),
        ],
        labels = [
            "Best model type",
            "All models",
        ]
    )
    plt.savefig("fig/double.pdf", dpi=300)
    plt.close()
