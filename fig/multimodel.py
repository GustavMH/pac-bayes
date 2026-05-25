#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("~/Downloads/pac-bayes-predictions/multimodel_eurosat.npz").expanduser()
#a = dict(np.load(path))

def calc_stats(val_preds, val_labels, test_preds, test_labels):
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
