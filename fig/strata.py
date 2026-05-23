#!/usr/bin/env python3

def calc_stats(val_preds, val_labels, test_preds, test_labels, n_iter_region=1):
    n_models, n_examples = val_preds.shape
    n_cats = 1+val_labels.max()
    uni = np.ones(n_models) / n_models

    risks, n1 = gibbs_risks(val_preds, val_labels)
    tnd, n2 = tandem_risks(val_preds, val_labels)
    params = {"tandem_risks": tnd, "n2": n2, "gibbs_risks": risks, "n1": n1}
    rho, bound, _ = bounds.optimize_rho("tnd", params)
    rho_best, _, _ = bounds.optimize_rho("best", params)
    rho_fo, bound_fo, _ = bounds.optimize_rho("lambda", params)

    return {
        **params,
        "tnd_rho": rho,
        "tnd_bound": bound,
        "tnd_test_loss": loss(rho, np.eye(n_cats)[test_preds], test_labels),
        "best_rho": rho_best,
        "fo_rho": rho_fo,
        "fo_bound": bound_fo,
        "fo_test_loss": loss(rho_fo, np.eye(n_cats)[test_preds], test_labels),
        "uni_test_loss": loss(uni, np.eye(n_cats)[test_preds], test_labels),
        "best_test_loss": loss(rho_best, np.eye(n_cats)[test_preds], test_labels),
        "uni_bound": loss(uni, np.eye(n_cats)[val_preds], val_labels) + np.sqrt(np.log(2/0.05)/(2*len(val_preds[0]))),
    }

def load_CIFAR10_strata():
    path = "~/Downloads/pac-bayes-predictions/cifar10_preds_fix.npz"
    path = Path(path).expanduser()
    res = dict(np.load(path))

    collect = []
    for i, n_models in product(range(30), range(2,16)):
        idx = np.flip(np.array([12, 13, 11, 10,  5,  0,  9,  4,  8,  7,  3,  1,  6,  2]))
        val_preds   = res["predictions_validation"][i, idx[:n_models]].argmax(-1)
        val_labels  = res["labels_validation"][i]
        test_preds  = res["predictions_test"][i, idx[:n_models]].argmax(-1)
        test_labels = res["labels_test"]

        collect.append(calc_stats(val_preds, val_labels, test_preds, test_labels))

    cifar10_strata = np.array(collect).reshape(30,14)

def load_CIFAR100_strata():
    pass

    path = "~/Downloads/pac-bayes-predictions/cifar100_preds_fix.npz"
    path = Path(path).expanduser()
    res = dict(np.load(path))

    collect = []
    for i, n_models in tqdm(list(product(range(30), range(2,16)))):
        idx = np.flip(np.array([12, 13, 11, 10,  5,  0,  9,  4,  8,  7,  3,  1,  6,  2]))
        val_preds   = res["predictions_validation"][i, idx[:n_models]].argmax(-1)
        val_labels  = res["labels_validation"][i]
        test_preds  = res["predictions_test"][i, idx[:n_models]].argmax(-1)
        test_labels = res["labels_test"]

        collect.append(calc_stats(val_preds, val_labels, test_preds, test_labels))

    cifar100_strata = np.array(collect).reshape(30,14)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

fig, axs = plt.subplots(1,3,figsize=(5.5,3))

def get2d(arr, name):
    x = np.array([[x[f"{name}_test_loss"] for x in y] for y in arr])
    µ = x.mean(0)
    std = x.std(0,ddof=1)
    return µ, std

for ax, risk, label in zip(axs, [imdb_strata, cifar10_strata, cifar100_strata], ["IMDB", "CIFAR10", "CIFAR100"]):
    ax.set_title(label)

    tnd = np.array([[x[f"tnd_test_loss"] for x in y] for y in risk])
    for name in ["fo", "tnd", "uni", "best"]:
        x = np.array([[x[f"{name}_test_loss"] for x in y] for y in risk])
        print(name, (utest(tnd, x).pvalue < 0.05 / (tnd.shape[1]*3))*1.0)

    for (µ, std) in [get2d(risk, key) for key in ["fo", "uni", "tnd", "best"]]:
        ax.plot(µ)
        ax.fill_between(np.arange(len(µ)), µ+std, µ-std, alpha=0.2)

    #ax.set_ylim(bottom=np.min(µ_tnd)-0.001, top=0.18)
    #ax.set_xticks(range(0,10,2),range(0,10,2))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))

from matplotlib.lines import Line2D

axs[0].set_ylim(bottom=0.139, top=0.2)
axs[1].set_ylim(top=0.0532)
axs[0].set_xticks([0,3,8],[2,5,10])
axs[1].set_xticks([0,3,8,13],[2,5,10,15])
axs[2].set_xticks([0,3,8,13],[2,5,10,15])
axs[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
axs[0].set_ylabel("Loss")
axs[1].set_xlabel("No. of ensemble members")
fig.legend(
    handles=[Line2D([],[],c=c) for c in ["tab:blue", "tab:orange", "tab:green", "tab:red"]],
    bbox_to_anchor=(0.5, 0.15),
    labels=["First-order", "Uniform", "Tandem", "Early Stopping"],
    loc="upper center",
    ncols=4
)
fig.tight_layout(rect=[0,0.1,1,1])
plt.savefig("fig/IMDB_strata.pdf")
plt.close()
