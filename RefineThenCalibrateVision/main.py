import os
import torch
import wandb
import pandas as pd
import torchvision.transforms as transforms
import numpy as np
import pickle

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import DataLoader, Subset



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


def main(args):
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch import Trainer, seed_everything

    from resnet import ResNet18
    from wide_resnet import WideResNet
    from utils import CalibrationRefinementTSModule, Cutout

    class CheckpointEval(Callback):
        def __init__(self, every_n_epochs, name, folder):
            self.every_n_epochs = every_n_epochs
            self.name = name
            self.folder = folder

        def on_validation_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch
            if (epoch + 1) % self.every_n_epochs == 0:
                folder = Path(self.folder)
                folder.mkdir(exist_ok=True)

                model = pl_module.model

                device = pl_module.device
                val_loader, test_loader = trainer.val_dataloaders

                test_pred = np.concatenate([model(X.to(device)).detach().cpu() for X, _ in test_loader])
                with open(self.folder / f"{self.name}_{epoch:02d}_test_predictions.pkl", "wb") as f:
                    pickle.dump(test_pred, f)

                val_pred = np.concatenate([model(X.to(device)).detach().cpu() for X, _ in val_loader])
                with open(self.folder / f"{self.name}_{epoch:02d}_val_predictions.pkl", "wb") as f:
                    pickle.dump(val_pred, f)

    train_set, val_set, test_set, num_classes = load_dataset(args.dataset)

    # We run for the specified number of seeds.
    for i in range(args.seeds_per_job):
        seed = args.seed + i
        seed_everything(seed, workers=True)

        # Logging config
        model_name = args.run_name + '_seed' + str(seed)
        save_dir = args.folder + args.dataset + '_' + args.model + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        wandb_logger = WandbLogger(
            project=args.project,
            save_dir=save_dir,
            name=model_name,
            log_model=False
        )
        for arg in vars(args):
            wandb_logger.experiment.config[arg] = getattr(args, arg)

        checkpointing = CheckpointEval(
            folder = Path(args.folder),
            name = f"{args.dataset}_{args.model}_run_{seed}"
        )

        # Lightning trainer
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=args.epochs,
            callbacks=[checkpointing]
        )

        # Perform a new train-val split for every seed
        train_indices, val_indices = train_test_split(list(range(len(train_set))), train_size=args.train_proportion, random_state=seed)
        train_set_ = Subset(train_set, train_indices)
        val_set_ = Subset(val_set, val_indices)

        train_loader = DataLoader(train_set_, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, shuffle=True)
        val_loader = DataLoader(val_set_, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, shuffle=False)

        # Instantiate model
        if args.model == 'resnet18':
            model = ResNet18(num_classes=num_classes)
        elif args.model == 'wideresnet':
            model = WideResNet(num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)
        else:
            raise Exception('Oops, requested model does not exist!')

        # Our refinement early stopping module
        lightning_module = CalibrationRefinementTSModule(model, args.lr_scheduler)

        # Training
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=[val_loader, test_loader],
        )

        # Logging run to csv for paper tables and figures
        logs = pd.DataFrame(lightning_module.logs)
        logs.to_csv(save_dir + args.lr_scheduler+'_'+model_name+'.csv')

        # Log a score table to wandb for the run to compare early stopping metrics
        comparison_table = wandb.Table(columns=[
            'stopping metric',
            'best epoch',
            'error',
            'auroc',
            'logloss',
            'brier',
            'ece',
            'smece'
        ])

        for (name, key, minimize) in [
            ('logloss', 'logloss/val', True),
            ('brier', 'brier/val', True),
            ('error', 'error/val_error', True),
            ('auroc', 'error/val_auroc', False),
            ('logloss TS', 'logloss/val_ts', True)
        ]:
            if minimize:
                best_epoch = logs[key].idxmin()
            else:
                best_epoch = logs[key].idxmax()
            best_epoch_metrics = logs.loc[best_epoch]
            comparison_table.add_data(
                name,
                best_epoch,
                best_epoch_metrics['error/test_error'],
                best_epoch_metrics['error/test_auroc'],
                best_epoch_metrics['logloss/test_ts'],
                best_epoch_metrics['brier/test_ts'],
                best_epoch_metrics['ece/test_ts'],
                best_epoch_metrics['smece/test_ts'],
            )

        wandb.log({'comparison_table': comparison_table})
        wandb.finish()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run_name', type=str, default='Unknown')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds_per_job', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_proportion', type=float, default=0.9)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--lr_scheduler', type=str, default='step')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--folder', type=str, default="./results")
    parser.add_argument('--project', type=str)
    if torch.cuda.is_available():
        parser.add_argument('--accelerator', type=str, default='gpu')
    else:
        parser.add_argument('--accelerator', type=str, default='cpu')
    args = parser.parse_args()

    main(args)
