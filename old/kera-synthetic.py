#!/usr/bin/env python3

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.datasets import make_moons

import keras
from keras.src.callbacks.callback import Callback


def plot_training_history(history, path="history.png"):
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.plot(1 - np.array(history.history["val_accuracy"]), label="1 - val_accuracy")
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_heatmap(X, y, model, path="heatmap.png"):
    m = 0.3
    res = 300
    axis1_extent = (X[:, 0].min() - m, m + X[:, 0].max())
    axis1 = np.linspace(*axis1_extent, res)
    axis2_extent = (X[:, 1].min() - m, m + X[:, 1].max())
    axis2 = np.linspace(*axis2_extent, res)
    points = np.array(np.meshgrid(axis1, axis2)).reshape(2, res**2).T
    predictions = np.array(model(points)).reshape(res, res)

    extent = (*axis1_extent, *axis2_extent)
    plt.imshow(predictions, extent=extent, origin="lower", cmap="bwr_r", alpha=0.2)
    # plt.contour(predictions, levels=[.5], colors='k', linestyles='-', extent=extent, alpha=.2)
    plt.scatter(*X[y == 0].T, c="red", s=1)
    plt.scatter(*X[y == 1].T, c="blue", s=1)
    plt.savefig(path)
    plt.close()


class CheckpointAndEval(Callback):
    def __init__(
        self,
        n_epochs,
        dest,
        name,
        dataset,
        log=lambda x: _,
        folder_path=None,
        plot=False,
    ):
        super().__init__()
        self.dest = dest
        self.n_epochs = n_epochs
        self.folder_path = folder_path
        self.name = name
        self.dataset = dataset
        self.log = log
        self.plot = plot

    def on_epoch_end(self, epoch, logs=None):
        e = epoch + 1

        if e % self.n_epochs == 0:
            self.log(f"Saving model trained for {e}")

            positive_pred = np.array(self.model(self.dataset["X_test"]))
            preds = np.array((positive_pred, 1-positive_pred)).squeeze().mT
            self.dest["predictions_test"][e // self.n_epochs - 1] = preds

            positive_pred = np.array(self.model(self.dataset["X_val"]))
            preds = np.array((positive_pred, 1-positive_pred)).squeeze().mT
            self.dest["predictions_validation"][e // self.n_epochs - 1] = preds

            if self.plot:
                plot_heatmap(
                    self.dataset["X_test"],
                    self.dataset["y_test"],
                    self.model,
                    Path(self.folder_path) / f"{self.name}_{e:02d}_test_heatmap.png",
                )


def train_model(
    dataset, dest, batch_size=32, n_epochs=1000, save_per_n_epochs=50, name="model"
):

    checkpointing = CheckpointAndEval(
        n_epochs=save_per_n_epochs,
        dest=dest,
        name=name,
        dataset=dataset,
        log=print,
    )

    model = keras.Sequential(
        [
            keras.layers.Dense(250, input_dim=2, activation="relu"),
            keras.layers.Dense(250, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        dataset["X_train"],
        dataset["y_train"],
        validation_data=(dataset["X_val"], dataset["y_val"]),
        callbacks=[checkpointing],
        epochs=n_epochs,
        verbose=0,
    )

    # plot_training_history(history, folder / f"{name}_history.png")

    return model


def main(n=None):
    if n == None:
        from argparse import ArgumentParser

        parser = ArgumentParser(
            prog="Train models that predict classes of the make_moons function"
        )
        parser.add_argument("-n", help="experiment number")
        experiment_n = parser.parse_args().n
        print(experiment_n, parser.parse_args())
    else:
        experiment_n = int(n)

    folder = Path(f"{experiment_n}_moon_models")
    folder.mkdir(exist_ok=True)

    n_train, n_test, n_val = 100, 1000, 100
    X, y = make_moons(
        n_samples=n_train + n_test + n_val, noise=0.5, random_state=int(experiment_n)
    )
    X_test, y_test = make_moons(
        n_samples=n_test, noise=0.1, random_state=int(experiment_n)
    )

    X_train, X_val = X[:n_train], X[-n_val:]
    y_train, y_val = y[:n_train], y[-n_val:]
    assert len(X_train) + len(X_test) + len(X_val) == len(X)

    dataset = {
        "X_train": X_train,
        "X_test": X_test,
        "X_val": X_val,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val,
    }

    with open(folder / "moon_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    results = {
        "predictions_test": np.zeros((10, 20, n_test, 2)),
        "predictions_validation": np.zeros((10, 20, n_val, 2))
    }
    for i in range(10):
        res = {
            "predictions_test": results["predictions_test"][i],
            "predictions_validation": results["predictions_validation"][i]
        }
        model = train_model(dataset, res, name=f"moon_model_{i}")

    return results

if __name__ == "__main__":
    main()
