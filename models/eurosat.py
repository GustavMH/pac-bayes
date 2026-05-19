#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from models.resnet import resnet18

from tqdm import tqdm

import numpy as np
from argparse import ArgumentParser
from itertools import batched, groupby, accumulate


def gpu_or_quit() -> str:
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
        print(f"Using {device}", flush=True)
        return device
    else:
        print("No GPU allocated >:(", flush=True)
        #exit(3)

device = gpu_or_quit()

def split_by_chroma(ds):
    imgs, labels = ds

    def chroma(img):
        return (img[0].mean() / img[1].mean()).item()

    c = [(y, chroma(X), i) for i, (X, y) in enumerate(zip(imgs, torch.argmax(labels,1)))]
    c = groupby(sorted(c), lambda x: x[0])
    idxs = [[i for y, c, i in grouper] for _, grouper in c]
    idx_A = torch.cat([torch.IntTensor(i[:len(i)//2]) for i in idxs])
    idx_B = torch.cat([torch.IntTensor(i[len(i)//2:]) for i in idxs])

    n = min(len(idx_A), len(idx_B))

    return idx_A[:n], idx_B[:n]

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

def train(model, dataset, scheduler, loss_fn, optimizer, n_epochs, callbacks=[], lr_step="epoch"):
    model.train()
    model.to(device)
    scaler = GradScaler()

    for epoch_n in tqdm(range(n_epochs)):
        running_loss = 0
        for batch_n, (X, y) in enumerate(ds_loader(dataset)):
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(X)
                loss = loss_fn(pred, y)

            running_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if lr_step == "batch":
                scheduler.step()

        if lr_step == "epoch":
            scheduler.step()

        for callback in callbacks:
            callback(model, epoch_n)

        print(f"Epoch {epoch_n}, Loss: {running_loss}")

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

def model_check():
    def _(model, epoch_n):
        for name, param in model.named_parameters():
            if "weight" in name:
                print(epoch_n, param[0,0])

    return _


def train_model(model, scheduler, ds, eval_sets, n_epochs=30):
    A_val, A_test, B_val, B_test = eval_sets
    print(A_val[0][0].shape)
    model(A_val[0][0].unsqueeze(0).to(device))
    print(
        "Training a "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):_}"
        f" parameter model, for {n_epochs} epochs"
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = scheduler(optimizer, cycle_size=args["cycle_size"])

    dest_test = torch.zeros((2, n_epochs, len(A_test[0]), 10))
    dest_val  = torch.zeros((2, n_epochs, len(A_val[0]),  10))

    train(
        model,
        ds,
        sched,
        loss_fn,
        optimizer,
        n_epochs,
        callbacks=[
            snapshot_callback(A_test, dest_test[0], 1),
            snapshot_callback(B_test, dest_test[1], 1),
            snapshot_callback(A_val, dest_val[0], 1),
            snapshot_callback(B_val, dest_val[1], 1),
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

def main(args=None):
    torch.cuda.empty_cache()

    if not args:
        parser = ArgumentParser(prog = "train on EuroSAT")
        parser.add_argument("-d","--dataset")
        parser.add_argument("--scheduler")
        parser.add_argument("--model")
        parser.add_argument("--out-file")
        parser.add_argument("--n-runs")
        parser.add_argument("--n-epochs")
        parser.add_argument("--cycle-size")

        args = parser.parse_args()

    ds = None
    match args["dataset"].lower():
        case "eurosat":
            ds = ImageFolder(
                "data/EuroSAT",
                transform=ToTensor(),
                target_transform=lambda x: torch.eye(10)[int(x)],
            )

    scheduler = None
    match args["scheduler"].lower():
        case "none":
            scheduler = torch.optim.lr_scheduler.LRScheduler
        case "cos":
            from models.schedulers import CyclicCosineAnnealingLR
            scheduler = CyclicCosineAnnealingLR
        case "tri":
            from models.schedulers import TriangularCyclicLR
            scheduler = TriangularCyclicLR

    model = None
    match args["model"].lower():
        case "resnet18":
            import models
            model = models.resnet.resnet18(weights=None)
        case "resnet18_pretrained":
            import models
            model = models.resnet.resnet18()
        case "mlp":
            import models.mlp
            model = models.mlp.MLP()

    assert(args["out_file"])
    assert(int(args["n_runs"]))
    assert(int(args["n_epochs"]))

    ds_labels = torch.vstack([y for _, y in tqdm(ds)]).to(device)
    ds_imgs = torch.cat([X.unsqueeze(0) for X, _ in tqdm(ds)]).to(device)
    ds = (ds_imgs, ds_labels)
    ds_num = torch.argmax(ds_labels, 1).to(device)

    idx_A, idx_B = split_by_chroma(ds)

    A_train, A_val, A_test = split_idx(idx_A, [0.7, 0.1, 0.2])
    B_train, B_val, B_test = split_idx(idx_B, [0.7, 0.1, 0.2])

    idx_mix = [[A_train, B_train] for _ in range(int(args["n_runs"]))]

    eval_sets = (ds_subset(ds, A_val), ds_subset(ds, A_test), ds_subset(ds, B_val), ds_subset(ds, B_test))

    res = [[train_model(model, scheduler, ds_subset(ds, idx), eval_sets, n_epocs=int(args["n_epochs"])) for idx in run] for run in idx_mix]
    res_val = np.array([[X["val"] for X in run] for run in res])
    res_test = np.array([[X["test"] for X in run] for run in res])

    try:
        # This should be different for the two val sets
        print((np.argmax(res_val, -1) == np.array(torch.index_select(ds_num, 0, A_val.to(device)).cpu())).mean(-1).round(1))
        print((np.argmax(res_val, -1) == np.array(torch.index_select(ds_num, 0, B_val.to(device)).cpu())).mean(-1).round(1))
    except:
        print("Woops!")

    np.savez_compressed(
        args["out_file"],
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

if __name__ == "__main__":
    main()
