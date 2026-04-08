#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

from torchvision.datasets import ImageFolder, folder
from torchvision.transforms import ToTensor, Compose

from pathlib import Path
from tqdm import tqdm

def one_hot(n_classes):
    def _(x):
        res = torch.zeros(n_classes)
        res[x] = 1
        return res
    return _

def load_AB(path, classes):
    def convert_class(m):
        return lambda x: classes.index(m[x])

    class_map = folder.find_classes(path)[1]
    class_map = {v: k for (k,v) in class_map.items()}

    return ImageFolder(
        path,
        transform=ToTensor(),
        target_transform=Compose([convert_class(class_map), one_hot(5)])
    )

A_path = Path("~/Downloads/EuroSAT_A").expanduser()
B_path = Path("~/Downloads/EuroSAT_B").expanduser()
A_classes = ['PermanentCrop', 'Forest',               'Industrial', 'Pasture',     'River']
B_classes = ['AnnualCrop',    'HerbaceousVegetation', 'Highway',    'Residential', 'SeaLake']
ds_A = load_AB(A_path, A_classes)
ds_B = load_AB(B_path, B_classes)

"AnnualCrop", "HerbaceousVegetation", "Highway", "Residential", "SeaLake"
A_train, A_val, A_test = torch.utils.data.random_split(ds_A, [.7, .1, .2])
B_train, B_val, B_test = torch.utils.data.random_split(ds_B, [.7, .1, .2])

def mix_subsets(A, B, ratio, n=None):
    # Switch to drawing N, with a specific ratio
    d = torch.utils.data
    n = n if n else min(len(subset_A), len(subset_B))
    subset_A, _ = d.random_split(A, [ratio, 1-ratio])
    _, subset_B = d.random_split(B, [ratio, 1-ratio])
    return d.ConcatDataset([subset_A, subset_B])

def MLP():
    return nn.Sequential(
        nn.Conv2d(3, 256, (3,3)),
        nn.LeakyReLU(),
        nn.Conv2d(8, 256, (3,3)),
        nn.Flatten(),
        nn.LeakyReLU(),
        nn.LazyLinear(128),
        nn.LeakyReLU(),
        nn.LazyLinear(64),
        nn.LeakyReLU(),
        nn.LazyLinear(5)
    )

def train(
        model,
        dataset,
        loss_fn,
        optimizer,
        n_epochs,
        callbacks=[]
):
    model.train()
    for epoch_n in tqdm(range(n_epochs)):
        for batch_n, (X, y) in enumerate(dataset):
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model

def test(model, dataset, loss_fn):
    res = torch.zeros(len(dataset))
    for i, (X, y) in enumerate(DataLoader(dataset, batch_size=1)):
        res[i] = loss_fn(model(X), y)
    return torch.mean(res)

def argmax_eq(y1, y2):
    return torch.argmax(y1) == torch.argmax(y2)

def train_mix(ratio):
    mlp = MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    mix = mix_subsets(A_train, B_train, .5)
    loader = DataLoader(mix, batch_size=64, shuffle=True)

    train(mlp, loader, loss_fn, optimizer, 1)
    return test(mlp, A_test, argmax_eq), test(mlp, B_test, argmax_eq)

res = torch.Tensor([train_mix(ratio) for ratio in [0.0, 0.25, 0.5, 0.75, 1]])
