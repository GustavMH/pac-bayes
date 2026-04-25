#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from tqdm import tqdm

import numpy as np

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
if device == "cpu":
    print("Allocated a cpu >:(", flush=True)
    exit(3)
else:
    print(f"Using {device}", flush=True)


def split_by_chroma(ds):
    from itertools import groupby

    def label(lst):
        return torch.argmax(lst[1])

    def chroma(img):
        return (img[0].mean() / img[1].mean()).item()

    chroma_by_label = groupby([(chroma(X), y) for X, y in tqdm(ds)], key=label)
    chroma_by_label = [[X for X, _ in group] for _, group in chroma_by_label]
    idx_by_label = [torch.argsort(torch.Tensor(lst)) for lst in chroma_by_label]
    idx_A = torch.cat([idxs[len(idxs) // 2 :] for idxs in idx_by_label])
    idx_B = torch.cat([idxs[: len(idxs) // 2] for idxs in idx_by_label])

    return Subset(ds, idx_A), Subset(ds, idx_B)


def mix_subsets(A, B, ratio, n=None):
    # Switch to drawing N, with a specific ratio
    d = torch.utils.data
    n = n if n else min(len(A), len(B))
    subset_A, _ = d.random_split(A, [ratio, 1 - ratio])
    _, subset_B = d.random_split(B, [ratio, 1 - ratio])
    return d.ConcatDataset([subset_A, subset_B])


def MLP():
    return nn.Sequential(
        nn.Conv2d(3, 64, (3, 3)),
        nn.LeakyReLU(),
        nn.Conv2d(64, 128, (3, 3)),
        nn.Flatten(),
        nn.LeakyReLU(),
        nn.LazyLinear(128),
        nn.LeakyReLU(),
        nn.LazyLinear(64),
        nn.LeakyReLU(),
        nn.LazyLinear(10),
    )


def train(model, dataset, loss_fn, optimizer, n_epochs, callbacks=[]):
    model.train()
    model.to(device)
    scaler = GradScaler()

    for epoch_n in tqdm(range(n_epochs)):
        for batch_n, (X, y) in enumerate(dataset):
            X, y = X.to(device), y.to(device)

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


def test(model, dataset, loss_fn):
    res = torch.zeros(len(dataset)).to(device)
    for i, (X, y) in enumerate(DataLoader(dataset, batch_size=1)):
        X, y = X.to(device), y.to(device)
        res[i] = loss_fn(model(X), y)
    return res.cpu()


def argmax_eq(y1, y2):
    return torch.argmax(y1) == torch.argmax(y2)


def snapshot_callback(dataset, dest, per_n_epochs):
    """Snapshot to DEST the labels the model would evaluate on DATASET PER_N_EPOCHS"""

    def save(model, epoch_n):
        if (epoch_n + 1) % per_n_epochs == 0:
            with torch.no_grad():
                res = dest[epoch_n // per_n_epochs]
                for i, (X, _) in enumerate(DataLoader(dataset, batch_size=1)):
                    X = X.to(device)
                    res[i] = model(X).cpu()

    return save

def train_mix(ratio, A, B):
    A_train, A_val, A_test = A
    B_train, B_val, B_test = B

    mlp = MLP().to(device)
    print(A_val[0][0].shape)
    mlp(A_val[0][0].unsqueeze(0).to(device))
    print(sum(p.numel() for p in mlp.parameters() if p.requires_grad))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    mix = mix_subsets(A_train, B_train, ratio)
    loader = DataLoader(mix, batch_size=64, shuffle=True)

    dest_test = torch.zeros((2, 15, len(A_test), 10))
    dest_val = torch.zeros((2, 15, len(A_val), 10))
    train(
        mlp,
        loader,
        loss_fn,
        optimizer,
        150,
        callbacks=[
            snapshot_callback(A_test, dest_test[0], 10),
            snapshot_callback(B_test, dest_test[1], 10),
            snapshot_callback(A_val, dest_val[0], 10),
            snapshot_callback(B_val, dest_val[1], 10),
        ],
    )
    return {"test": dest_test, "val": dest_val}

torch.cuda.memory._record_memory_history(max_entries=100000)

torch.cuda.empty_cache()

ds = ImageFolder(
    "data/EuroSAT",
    transform=ToTensor(),
    target_transform=lambda x: torch.eye(10)[int(x)],
)

ds_A, ds_B = split_by_chroma(ds)

A = torch.utils.data.random_split(ds_A, [0.7, 0.1, 0.2])
B = torch.utils.data.random_split(ds_B, [0.7, 0.1, 0.2])

res = [train_mix(ratio, A, B) for ratio in [0.0, 0.25, 0.5, 0.75, 1]]
res_val = np.array([X["val"] for X in res])
res_test = np.array([X["test"] for X in res])
np.savez_compressed(
    "models/eurosat_chroma_shift.npz", validation=res_val, test=res_test
)
