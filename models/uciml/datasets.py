from uciml.uciml_fetch import fetch_ucirepo, list_available_datasets
import torch
import numpy as np
import math


def isnan_obj_arr(arr):
    for c in arr:
        if isinstance(c, float) and math.isnan(c):
            return True
    return False


def load_dataset(ds_name):
    ds = fetch_ucirepo(name=ds_name)
    match ds_name:
        case "Contraceptive Method Choice":
            res = np.zeros((1473, 12 + 5))
            pos = 0
            o_keys = ["wife_edu", "husband_edu", "husband_occupation", "standard_of_living_index"]
            for key in o_keys:
                cat_char = ds.data.features[key].unique()
                cat_map = lambda c: dict((y, x) for x, y in enumerate(cat_char))[c]
                cat_idx = [cat_map(c) for c in ds.data.features[key]]
                cat_hot = np.eye(np.max(cat_idx) + 1)[cat_idx]
                # Remove one category if its redundant
                cat_hot = cat_hot[:, (0 if isnan_obj_arr(cat_char) else 1):]
                res[:, pos:pos+cat_hot.shape[1]] = cat_hot
                pos += cat_hot.shape[1]

            c_keys = ["wife_age", "num_children", "wife_religion", "wife_working", "media_exposure"]
            pos = 12
            for key in c_keys:
                res[:, pos] = np.array(ds.data.features[key], dtype=np.float64)
                nan_idx = np.isnan(res[:, pos])
                if np.any(nan_idx):
                    res[nan_idx, pos] = 0
                    pos += 1
                    res[:, pos] = nan_idx
                pos += 1

            X = torch.tensor(res, dtype=torch.float)
            y = ds.data.targets.to_numpy().squeeze() > 1
            y = torch.tensor(y, dtype=torch.float).unsqueeze(1)

            return X, y

        case "Heart Disease":
            X = torch.tensor(ds.data.features.to_numpy(), dtype=torch.float)
            X = torch.nan_to_num(X) # Handle missing entries by zero imputation
            y = torch.tensor(ds.data.targets.to_numpy(), dtype=torch.float)
            y[y > 0] = 1 # Turn into binary classification task

            return X, y

        case "Credit Approval":
            # One hot encode category inputs
            res = np.zeros((690, 45 + 8))
            pos = 0
            o_keys = ["A13", "A12", "A10", "A9", "A7", "A6", "A5", "A4", "A1"]
            for key in o_keys:
                cat_char = ds.data.features[key].unique()
                cat_map = lambda c: dict((y, x) for x, y in enumerate(cat_char))[c]
                cat_idx = [cat_map(c) for c in ds.data.features[key]]
                cat_hot = np.eye(np.max(cat_idx) + 1)[cat_idx]
                # Remove one category if its redundant
                cat_hot = cat_hot[:, (0 if isnan_obj_arr(cat_char) else 1):]
                res[:, pos:pos+cat_hot.shape[1]] = cat_hot
                pos += cat_hot.shape[1]

            c_keys = ["A15", "A14", "A11", "A8", "A3", "A2"]
            pos = 42
            for key in c_keys:
                res[:, pos] = np.array(ds.data.features[key], dtype=np.float64)
                nan_idx = np.isnan(res[:, pos])
                if np.any(nan_idx):
                    res[nan_idx, pos] = 0
                    pos += 1
                    res[:, pos] = nan_idx
                pos += 1

            X = torch.tensor(res, dtype=torch.float)
            y = ds.data.targets.to_numpy().squeeze()
            cat_map = lambda c: {"-": 0, "+": 1}[c]
            cat_idx = np.array([cat_map(c) for c in y])[:, None]
            y = torch.tensor(cat_idx, dtype=torch.float)

            return X, y
