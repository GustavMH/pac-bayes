#!/usr/bin/env python3
import re
import pickle
from pathlib import Path
import numpy as np

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


def select_indicies(folder, mode="all", dataset="dataset.pkl"):
    """ Select a subset of model checkpoints from a folder """
    candidate_strs = Path(folder).glob("*.h5")
    candidate_strs = [p.name for p in candidate_strs]
    candidate_strs = sorted(candidate_strs)

    def model_to_val_file(name):
        base = re.match("(.+?).weights.h5", name)[1]
        path = folder / f"{base}_val_predictions.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def best_model_idx(preds, truth):
        """
        Given a sequence of predictions, evaluate for
        accuracy against y_true, return the best
        """
        accs = [
            sum([
                (t > .5) == (p > .5)
                for (t,p) in zip(truth, pred)
            ]) / len(pred) for pred in preds
        ]
        return accs.index(max(accs))

    match mode:
        case "all":
            return [s for (s, _, _) in candidate_strs]
        case "best":
            y_val = pickle.load(open(folder / dataset, "rb"))["y_val"]

            candidate_strs = group_strs(candidate_strs)
            val_preds = [[model_to_val_file(s) for s in ss] for ss in candidate_strs]
            best_idxs = [best_model_idx(model_preds, y_val) for model_preds in val_preds]

            offsets = np.arange(len(best_idxs)) * len(val_preds[0])
            return offsets + np.array(best_idxs)

        case "last":
            candidate_strs = group_strs(candidate_strs)
            n_models, n_checkpoints = np.array(candidate_strs).shape
            return np.arange(1,n_models+1) * n_checkpoints - 1

def ensemble_predictions(preds, weights):
    return np.array(preds.flat) * weights

def generate_rho_weights():

select_indicies(Path("moon_models"), "last")
