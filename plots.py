#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ensemble import ensemble_predictions, select_model_predictions
from majority_vote_bounds import optimize_rho

def moon_plot(select="best checkpoints"):
    folder = Path("0_moon_models")
    n_cats = 2
    dataset = pickle.load(open(folder / "moon_dataset.pkl", "rb"))
    y_val = dataset["y_val"]
    y_test = dataset["y_test"]
    preds_val, preds_test = select_model_predictions(folder, select, y_val=y_val)
    preds_models_val = (preds_val.squeeze() > 0.5).astype(np.int64)
    preds_models_test = (preds_test.squeeze() > 0.5).astype(np.int64)

    def make_rho_ensemble(subset_idx):
        bound, rho, lam = optimize_rho(preds_models_val[subset_idx], y_val)
        preds_ensemble_test = ensemble_predictions(rho, preds_models_test[subset_idx], n_cats)
        return (preds_ensemble_test == y_test).mean()

    def make_uniform_ensemble(subset_idx):
        rho = np.ones(subset_idx.size) / subset_idx.size
        preds_ensemble_test = ensemble_predictions(rho, preds_models_test[subset_idx], n_cats)
        return (preds_ensemble_test == y_test).mean()

    def make_early_stopping(subset_idx):
        val_scores = (preds_models_val[subset_idx] == y_val).mean(1)
        preds_test = preds_models_test[val_scores.argmax()]
        return (preds_test == y_test).mean()


    X = np.arange(1,31)
    early_acc   = np.zeros(X.size)
    rho_acc     = np.zeros(X.size)
    uniform_acc = np.zeros(X.size)

    n_iter = 25
    for _ in range(n_iter):
        subsets = [np.random.choice(np.arange(30), i) for i in X]
        #subsets = [np.arange(i) for i in X]
        early_acc   += [make_early_stopping(subset) for subset in subsets]
        rho_acc     += [make_rho_ensemble(subset) for subset in subsets]
        uniform_acc += [make_uniform_ensemble(subset) for subset in subsets]

    early_acc   /= n_iter
    rho_acc     /= n_iter
    uniform_acc /= n_iter

    plt.title(f"Ensembles of '{select}', on make_moons()")
    plt.plot(X, rho_acc, label="PAC-Bayes ensemble")
    plt.plot(X, uniform_acc, label="Uniform ensemble")
    plt.plot(X, early_acc, label="Best model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("No. of models")
    plt.ylim(.8,1)
    plt.legend()
    plt.savefig("ensemble.png")
    plt.close()


def plot_cifar10():
    subsets = [np.arange(0,15,i) for i in range(1,6)]
    #subsets = [np.arange(0,i) for i in range(1,16)]
    X = np.array([s.size for s in subsets])
    early_acc   = np.zeros(X.size)
    rho_acc     = np.zeros(X.size)
    uniform_acc = np.zeros(X.size)
    train_set, val_set, test_set, n_cats = load_dataset("cifar10")

    for i in range(30):
        seed = 10 + i
        folder = Path("RefineThenCalibrateVision") / "cifar10_models"

        _, val_indices = train_test_split(np.arange(len(train_set)), train_size=.9, random_state=seed)
        val_set_ = Subset(val_set, val_indices)
        X_val = np.array([X for X, _ in val_set_])
        y_val = np.array([y for _, y in val_set_])
        y_test = np.array([y for _, y in test_set])

        select = "single run"
        preds_val, preds_test = select_model_predictions(folder, select, regex="(\d+)_(\d+)_val_predictions.pkl", run_idx=i)
        preds_models_val = preds_val
        preds_models_test = preds_test

        def make_rho_ensemble(subset_idx):
            bound, rho, lam = optimize_rho(preds_models_val.argmax(axis=2)[subset_idx], y_val)
            #print(rho)
            preds_ensemble_test = ensemble_predictions_AVG(rho, preds_models_test[subset_idx], n_cats)
            return (preds_ensemble_test == y_test).mean()

        def make_uniform_ensemble(subset_idx):
            rho = np.ones(subset_idx.size) / subset_idx.size
            preds_ensemble_test = ensemble_predictions_AVG(rho, preds_models_test[subset_idx], n_cats)
            return (preds_ensemble_test == y_test).mean()

        def make_early_stopping(subset_idx):
            val_scores = (preds_models_val.argmax(axis=2)[subset_idx] == y_val).mean(1)
            preds_test = preds_models_test.argmax(axis=2)[val_scores.argmax()]
            return (preds_test == y_test).mean()

        # Select different epoch spacing
        early_acc   += [make_early_stopping(subset) for subset in subsets]
        rho_acc     += [make_rho_ensemble(subset) for subset in subsets]
        uniform_acc += [make_uniform_ensemble(subset) for subset in subsets]

    early_acc   /= 30
    rho_acc     /= 30
    uniform_acc /= 30

    plt.title(f"Ensembles of '{select}', on CIFAR10")
    plt.plot(X, rho_acc, label="PAC-Bayes ensemble")
    plt.plot(X, uniform_acc, label="Uniform ensemble")
    plt.plot(X, early_acc, label="Best model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("No. of models")
    #plt.xscale("log")
    plt.ylim(.95,.957)
    plt.legend()
    plt.savefig("ensemble.png")
    plt.close()
