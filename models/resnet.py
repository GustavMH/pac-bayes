#!/usr/bin/env python3
from torchvision.models import resnet18

def resnet18_n_cat(n_cats=10):
    model = resnet18("IMAGENET1K_V1")
    model.fc = nn.Linear(mlp.fc.in_features, n_cats)
