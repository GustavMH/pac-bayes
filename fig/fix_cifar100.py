#!/usr/bin/env python3
import numpy as np
from pathlib import Path

def fix_CIFAR(res):

    val_preds = res["predictions_validation"]
    val_index = np.isnan(val_preds[:,0,:,0])
    val_index = np.array([np.arange(len(nans))[~nans] for nans in val_index])
    val_preds_sub = np.array([val_preds[i,:,val_index[i]] for i in range(len(val_index))]).transpose((0,2,1,3))

    np.savez_compressed(
        Path("~/cifar100_preds_fix.npz").expanduser(),
        labels_validation = res["labels_validation"].astype(np.uint8),
        predictions_validation = val_preds_sub.astype(np.float16),
        labels_test = res["labels_test"].astype(np.uint8),
        predictions_test = res["predictions_test"].astype(np.float16),
        val_index = val_index
    )

path = "~/cifar100_predictions.npz"
path = Path(path).expanduser()
res = dict(np.load(path))

fix_CIFAR(res)
