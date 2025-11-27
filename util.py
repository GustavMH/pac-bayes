#!/usr/bin/env python3
import numpy as np
import pandas as pd
from itertools import islice

def oob_tandem_risks(
        preds: [np.array],
        pred_idx: [np.array],
        targs: np.array
):
    """
    Calculate a (model, model) size array of risk scores.

    Predictions from an out-of-bag (unused for optimization) validation
    set; the PREDS array has shape (model, prediction), the TARGS array
    contains the dataset from which the validation set has been drawn
    """
    assert(len(preds) == len(pred_idx))
    assert(all(p.shape == i.shape for (p,i) in zip(preds, pred_idx)))

    m = len(preds)
    tandem_risks = np.zeros((m,m))
    n_intersects = np.zeros((m,m))

    for i, (preds_a, idx_a) in enumerate(zip(preds, pred_idx)):
        for j, (preds_b, idx_b) in islice(enumerate(zip(preds, pred_idx)), i, m):
            idxs, c_a, c_b = np.intersect1d(idx_a, idx_b, True, True)

            risk = np.sum(
                np.logical_and(
                    preds_a[c_a] != targs[idxs],
                    preds_b[c_b] != targs[idxs]
                )
            )
            tandem_risks[i,j] = risk
            tandem_risks[j,i] = risk

            n = len(idxs)
            n_intersects[i,j] = n
            n_intersects[j,i] = n

    return tandem_risks / n_intersects, n_intersects.min()



def oob_gibbs_risks(preds, pred_idx, targs):
    """
    Calculate a (model) size array of risk scores.

    Predictions from an out-of-bag (unused for optimization) validation
    set; the PREDS array has shape (model, prediction), the TARGS array
    contains the dataset from which the validation set has been drawn
    """
    assert(len(preds) == len(pred_idx))
    assert(all(p.shape == i.shape for (p,i) in zip(preds, pred_idx)))

    m = len(preds)
    risks = np.zeros(m)
    n_intersects = np.zeros(m)

    for i, (preds, idxs) in enumerate(zip(preds, pred_idx)):
        risks[i] = np.sum(preds != targs[idxs])
        n_intersects[i] = len(idxs)

    return risks / n_intersects, n_intersects.min()


def split_bootstrap(
        rng: np.random.RandomState,
        X: np.array,
        Y: np.array,
        n_samples: int,
        n_classes: None | int
) -> ((np.array, np.array), (np.array, np.array)):
    """
    Sample points (x,y) in (X,Y) w. replacement, get at least one example per class in Y.
    Return the sample and sample indicies and out-of-bag samples and indicies
    """

    n_classes = n_classes if (n_classes is None) else len(np.unique(Y))
    Y_sample = np.zeros((n_samples, *Y.shape[1:]))

    while not len(np.unique(Y_sample)) == n_classes:
        sample_idx = rng.randint(n_samples, size=n_samples)
        Y_sample = Y[sample_idx]

    oob_idx = np.delete(np.arange(len(X)), sample_idx)

    return sample_idx, oob_idx


def arr2d_to_df(
        arr: np.array,
        col_names: [str],
        row_names: [str]
) -> pd.DataFrame:
    """
    Convert a 2d numpy arr into a pandas DataFrame w. row and column names.
    """

    assert(len(arr.shape) == 2)
    #assert(len(row_names) == arr.shape[0])
    #assert(len(col_names) == arr.shape[1])

    return pd.DataFrame({
        "name": row_names,
        **dict([
            (name, arr[i])
            for i, name in enumerate(col_names)
        ])
    })
