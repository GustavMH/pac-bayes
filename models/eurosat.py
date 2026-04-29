#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.models import resnet18

from tqdm import tqdm

import numpy as np
from itertools import batched, groupby, accumulate

def gpu_or_quit() -> str:
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
        print(f"Using {device}", flush=True)
        return device
    else:
        print("No GPU allocated >:(", flush=True)
        #exit(3)

def split_by_chroma(ds):
    imgs, labels = ds

    def chroma(img):
        return (img[0].mean() / img[1].mean()).item()

    c = [(y, chroma(X), i) for i, (X, y) in enumerate(zip(imgs, torch.argmax(labels,1)))]
    c = groupby(sorted(c), lambda x: x[0])
    idxs = [[i for y, c, i in grouper] for _, grouper in c]
    idx_A = torch.cat([torch.IntTensor(i[:len(i)//2]) for i in idxs])
    idx_B = torch.cat([torch.IntTensor(i[len(i)//2:]) for i in idxs])

    return idx_A, idx_B

def mix_idxs(idx_A, idx_B, ratio, n=None):
    # Switch to drawing N, with a specific ratio
    n = n if n else min(len(idx_A), len(idx_B))
    perm = torch.randperm(n)
    idx = torch.cat([idx_A[perm[int(n * ratio):]],
                     idx_B[perm[:int(n * ratio)]]])
    assert len(idx == n)
    return idx

def split_idx(idx, ratios):
    perm = torch.randperm(len(idx))
    lens = [int(len(idx)*r) for r in ratios]
    return [
        idx[perm[offset - n : offset]]
        for offset, n in zip(accumulate(lens), lens)
    ]

def train(model, dataset, loss_fn, optimizer, n_epochs, callbacks=[]):
    model.train()
    model.to(device)
    scaler = GradScaler()

    for epoch_n in tqdm(range(n_epochs)):
        for batch_n, (X, y) in enumerate(dataset):

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(X)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        for callback in callbacks:
            callback(model, epoch_n)

    return model


def snapshot_callback(ds, dest, per_n_epochs, batch_size=64):
    """Snapshot to DEST the labels the model would evaluate on DATASET PER_N_EPOCHS"""

    def save(model, epoch_n):
        if (epoch_n + 1) % per_n_epochs == 0:
            with torch.no_grad():
                res = dest[epoch_n // per_n_epochs]
                for i, (X,_) in enumerate(ds_loader(ds, batch_size, False)):
                    X = X.to(device)
                    res[i*batch_size:(i+1)*batch_size] = model(X).cpu()

    return save


def train_resnet18(ds, eval_sets, n_epochs=150):

    A_val, A_test, B_val, B_test = eval_sets
    mlp = resnet18("IMAGENET1K_V1").to(device)
    mlp.fc = nn.Linear(mlp.fc.in_features, 10).to(device)
    print(A_val[0][0].shape)
    mlp(A_val[0][0].unsqueeze(0).to(device))
    print(
        "Training a "
        f"{sum(p.numel() for p in mlp.parameters() if p.requires_grad):_}"
        f" parameter model, for {n_epochs} epochs"
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    loader = ds_loader(ds)

    dest_test = torch.zeros((2, n_epochs // 10, len(A_test[0]), 10))
    dest_val  = torch.zeros((2, n_epochs // 10, len(A_val[0]),  10))

    train(
        mlp,
        loader,
        loss_fn,
        optimizer,
        n_epochs,
        callbacks=[
            snapshot_callback(A_test, dest_test[0], 10),
            snapshot_callback(B_test, dest_test[1], 10),
            snapshot_callback(A_val, dest_val[0], 10),
            snapshot_callback(B_val, dest_val[1], 10),
        ],
    )
    return {"test": dest_test, "val": dest_val}


def ds_subset(ds, idx):
    ds_imgs, ds_labels = ds
    imgs = torch.index_select(ds_imgs, 0, idx.to(device))
    labels = torch.index_select(ds_labels, 0, idx.to(device))
    return (imgs, labels)

def ds_loader(ds, batch_size=64, shuffle=True):
    imgs, labels = ds
    perm = torch.randperm(len(imgs)) if shuffle else torch.arange(len(imgs))
    perm = perm.to(device)
    for i in range(0, len(imgs), batch_size):
        batch = perm[i : i + batch_size]
        yield (
            torch.index_select(imgs, 0, batch),
            torch.index_select(labels, 0, batch),
        )

torch.cuda.empty_cache()

device = gpu_or_quit()
ds = ImageFolder(
    "data/EuroSAT",
    transform=ToTensor(),
    target_transform=lambda x: torch.eye(10)[int(x)],
)
ds_labels = torch.vstack([y for _, y in tqdm(ds)]).to(device)
ds_imgs = torch.cat([X.unsqueeze(0) for X, _ in tqdm(ds)]).to(device)
ds = (ds_imgs, ds_labels)
ds_num = torch.argmax(ds_labels, 1).to(device)

idx_A, idx_B = split_by_chroma(ds)

A_train, A_val, A_test = split_idx(idx_A, [0.7, 0.1, 0.2])
B_train, B_val, B_test = split_idx(idx_B, [0.7, 0.1, 0.2])

idx_mix = [[mix_idxs(A_train, B_train, ratio) for ratio in [0.0, 0.25, 0.5, 0.75, 1]] for _ in range(10)]

eval_sets = (ds_subset(ds, A_val), ds_subset(ds, A_test), ds_subset(ds, B_val), ds_subset(ds, B_test))

res = [[train_resnet18(ds_subset(ds, idx), eval_sets) for idx in run] for run in idx_mix]
res_val = np.array([[X["val"] for X in run] for run in res])
res_test = np.array([[X["test"] for X in run] for run in res])

try:
    # This should be different for the two val sets
    print((np.argmax(res_val, -1) == np.array(torch.index_select(ds_num, 0, A_val.to(device)).cpu())).mean(-1).round(1))
    print((np.argmax(res_val, -1) == np.array(torch.index_select(ds_num, 0, B_val.to(device)).cpu())).mean(-1).round(1))
except:
    print("Woops!")

np.savez_compressed(
    "models/eurosat_chroma_shift.npz",
    validation=res_val,
    test=res_test,
    val_labels=np.array(
        [
            torch.index_select(ds_num, 0, A_val.to(device)).cpu(),
            torch.index_select(ds_num, 0, B_val.to(device)).cpu(),
        ]
    ),
    test_labels=np.array(
        [
            torch.index_select(ds_num, 0, A_test.to(device)).cpu(),
            torch.index_select(ds_num, 0, B_test.to(device)).cpu(),
        ]
    ),
)
