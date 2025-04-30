#!/usr/bin/env python3
import re
import pickle
from pathlib import Path
import numpy as np

from majority_vote_bounds import optimize_rho

def group_strs(strs):
    """Groups identity strs by the first number in the string"""

    ss = [(s, re.search("\d+", s)[0]) for s in strs]
    ss = sorted(ss, key=lambda x: int(x[1]))

    res = []
    acc = []
    current_n = "0"
    for s, n in ss:
        if n == current_n:
            acc.append(s)
        else:
            current_n = n
            res.append(acc)
            acc = []
            acc.append(s)
    res.append(acc)

    return np.array(res)


def select_model_predictions(folder, mode="all", dataset="dataset.pkl"):
    """ Select a subset of model checkpoints from a folder """
    candidate_strs = Path(folder).glob("*.h5")
    candidate_strs = [p.name for p in candidate_strs]
    candidate_strs = sorted(candidate_strs)

    def model_to_val_file(name):
        base = re.match("(.+?).weights.h5", name)[1]
        path = folder / f"{base}_val_predictions.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def best_pred(preds, truth):
        """
        Given a sequence of predictions, evaluate for
        accuracy against y_true, return the best
        """
        return max([
            (pred,
             sum([
                 (t > .5) == (p > .5)
                 for (t,p) in zip(truth, pred)
             ]) / len(pred))
            for pred in preds
        ], key=lambda x: x[1])[0]

    match mode:
        case "all":
            return np.array([model_to_val_file(s) for s in candidate_strs])
        case "best":
            y_val = pickle.load(open(folder / dataset, "rb"))["y_val"]

            candidate_strs = group_strs(candidate_strs)
            val_preds = [[model_to_val_file(s) for s in ss] for ss in candidate_strs]
            best_preds = [best_pred(model_preds, y_val) for model_preds in val_preds]
            return np.array(best_preds)

        case "last":
            candidate_strs = group_strs(candidate_strs)
            last_strs = [model[-1] for model in candidate_strs]
            return np.array([model_to_val_file(s) for s in last_strs])

def ensemble_predictions(predictions, target):
    bound, rho, lam = optimize_rho(predictions, target)
    # TODO argmax
    return (predictions * rho[:,None]).mean(axis=0) > .5

reload(majority_vote_bounds)
folder = Path("moon_models")
y_val = pickle.load(open(folder / "dataset.pkl", "rb"))["y_val"]
preds_models = select_model_predictions(folder, "best")
preds_models = (preds_models.squeeze() > .5).astype(np.float64)
ensemble_predictions(preds_models, y_val)
