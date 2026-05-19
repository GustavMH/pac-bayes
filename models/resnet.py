#!/usr/bin/env python3

from torchvision.models import resnet18 as RN18
import torch.nn as nn

def resnet18(n_cats=10, weights="IMAGENET1K_V1"):
    mlp = RN18(weights="IMAGENET1K_V1")
    mlp.fc = nn.Linear(mlp.fc.in_features, n_cats)
    return mlp
