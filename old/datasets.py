#!/usr/bin/env python3
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
import torchvision.transforms as transforms

def load_dataset(dataset):
    from utils import Cutout

    # Training set transforms
    random_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16)
    ])
    # Validation/test transforms
    fixed_transform = transforms.Compose([
        transforms.ToTensor()
    ])


    data_folder = "/scratch/zvq211_data" if Path("/scratch").exists() else "./data"

    if dataset == 'cifar10':
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )
        random_transform.transforms.append(normalize)
        fixed_transform.transforms.append(normalize)
        train_set = CIFAR10(root=data_folder, train=True, transform=random_transform, download=True)
        val_set = CIFAR10(root=data_folder, train=True, transform=fixed_transform, download=True)
        test_set = CIFAR10(root=data_folder, train=False, transform=fixed_transform, download=True)
        num_classes = 10

    elif dataset == 'cifar100':
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )
        random_transform.transforms.append(normalize)
        fixed_transform.transforms.append(normalize)
        train_set = CIFAR100(root=data_folder, train=True, transform=random_transform, download=True)
        val_set = CIFAR100(root=data_folder, train=True, transform=fixed_transform, download=True)
        test_set = CIFAR100(root=data_folder, train=False, transform=fixed_transform, download=True)
        num_classes = 100

    elif dataset == 'svhn':
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
            std=[x / 255.0 for x in [50.1, 50.6, 50.8]]
        )
        random_transform.transforms.append(normalize)
        fixed_transform.transforms.append(normalize)
        train_set = SVHN(root=data_folder, split='train', transform=random_transform, download=True)
        val_set = SVHN(root=data_folder, split='train', transform=fixed_transform, download=True)
        test_set = SVHN(root=data_folder, split='test', transform=fixed_transform, download=True)
        num_classes = 10

    else:
        raise Exception('Oops, requested dataset does not exist!')

    return train_set, val_set, test_set, num_classes
