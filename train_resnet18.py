import wandb
import numpy as np

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer, seed_everything
from lightning import LightningModule

from wide_resnet import WideResNet

from dataset import load_dataset

class CheckpointEval(Callback):
    def __init__(self, loader, dest, idxs):
        self.dest = dest
        self.loader = loader
        self.idxs = idxs

    def on_validation_epoch_end(self, trainer, pl_module):
        model = pl_module.model
        device = pl_module.device
        epoch = trainer.current_epoch

        pred = np.concatenate([model(X.to(device)).detach().cpu() for X, _ in loader])
        dest[epoch, self.idxs] = pred

class ResNet18_cos_cycle(LightningModule):
    def __init__(self, cycle_n_epochs, batch_size):
        from torchvision.models import resnet18
        self.model = resnet18()
        self.loss = nn.CrossEntropyLoss()
        # HACK avoids custom optimizer_step()
        self.step_between_cycle = cycle_n_epochs * batch_size

    def forward(self, batch):
        inputs, _ = batch
        with torch.no_grad():
            return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        output = self.model(inputs)
        loss = self.loss(targets, outputs)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1**-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.step_between_cycle)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }


def main(args):
    train_set, val_set, test_set, num_classes = load_dataset(args.dataset)

    res_val  = np.full((n_models, n_epochs, len(val_set),  n_cats), np.nan)
    res_test = np.full((n_models, n_epochs, len(test_set), n_cats), np.nan)

    for i in range(args.seeds_per_job):
        seed = args.seed + i
        seed_everything(seed, workers=True)

        # Logging config
        model_name = f"{args.run_name}_seed_{seed}"
        save_dir = f"{args.folder}_{args.dataset}_{args.model}/"
        Path(save_dir).mkdir(exists_ok=True)

        wandb_logger = WandbLogger(
            save_dir=save_dir,
            name=model_name,
            log_model=False
        )

        for arg in vars(args):
            wandb_logger.experiment.config[arg] = getattr(args, arg)

        # Perform a new train-val split for every seed
        train_indices, val_indices = train_test_split(list(range(len(train_set))), train_size=args.train_proportion, random_state=seed)
        train_set_ = Subset(train_set, train_indices)
        val_set_ = Subset(val_set, val_indices)

        train_loader = DataLoader(train_set_, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, shuffle=True)
        val_loader = DataLoader(val_set_, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, shuffle=False)

        checkpointing_test = CheckpointEval(test_loader, res_test[i])
        checkpointing_val  = CheckpointEval(val_loader,  res_val [i])

        # Lightning trainer
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=args.epochs,
            callbacks=[
                checkpointing_test,
                checkpointing_val
            ]
        )

        # Training
        trainer.fit(
            model=ResNet18_cos_cycle(50, args.batch_size),
            train_dataloaders=train_loader,
            val_dataloaders=[val_loader, test_loader],
        )

        wandb.finish()

    np.savez_compressed(
        f"{args.dataset}_predictions.npz",
        predictions_validation=res_val,
        predictions_test=res_test,
        description=np.array(f"{args.dataset} aggregate predictions in the shape; " \
                            "Models, Checkpoints, Predictions, Categories. " \
                            "Checkpoints where taken every epoch"),
        labels_validation = val_set.squeeze(),
        labels_test = test_set.squeeze()
    )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run_name', type=str, default='Unknown')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds_per_job', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_proportion', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--folder', type=str, default="./results")
    args = parser.parse_args()

    main(args)
