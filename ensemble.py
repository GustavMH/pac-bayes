#!/usr/bin/env python3
import pickle
from pathlib import Path

def select_indicies(folder, mode="all"):
    """ Select a subset of model checkpoints from a folder """
    candidate_strs = Path(folder).glob("*.h5")
    candidate_strs = [p.name for p in candidate_strs]
    candidate_strs = sorted(candidate_strs)

    def model_to_val_file(name):
        base = re.match("(.+?).weights.h5", name)[1]
        path = folder / f"{base}_val_predictions.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    match mode:
        case "all":
            return [s for (s, _, _) in candidate_strs]
        case "best":
            y_val = 0
            candidate_vals = [model_to_val_file(s) for s in candidate_strs]
            # stack, compare for accuracy, mean axis=1, argmax
        case "last":
            pass

select_indicies(Path("moon_models"), "best")
