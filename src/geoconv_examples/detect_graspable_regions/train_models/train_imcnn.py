from geoconv_examples.detect_graspable_regions.data.dataset import PartNetDataset
from geoconv_examples.detect_graspable_regions.models.imcnn import SegImcnn

from torch import nn
from matplotlib import pyplot as plt

import pandas as pd
import torch
import os


def log_training(model, hist, logging_dir, verbose=True):
    """Saves model, training history and plots training statistics."""
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Save model
    torch.save(model.state_dict(), f"{logging_dir}/model.zip")

    # Split train/val from test
    train_val_df = pd.DataFrame.from_dict({
        "train_loss": hist["train_loss"],
        "train_accuracy": hist["train_accuracy"],
        "val_loss": hist["val_loss"],
        "val_accuracy": hist["val_accuracy"],
    })
    test_df = pd.DataFrame.from_dict({
        "test_loss": hist["test_loss"],
        "test_accuracy": hist["test_accuracy"]
    })

    # Save csv files
    train_val_df.to_csv(f"{logging_dir}/train_val.csv", index=False)
    test_df.to_csv(f"{logging_dir}/test.csv", index=False)

    # Save plot
    n_rows, n_cols, cm = 2, 1, 1 / 2.54
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12 * cm, 12.2 * cm), sharex=True)
    axs[0].set_ylabel("Loss")
    train_val_df.plot(y=["train_loss", "val_loss"], ax=axs[0], grid=True)
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epoch")
    train_val_df.plot(y=["train_accuracy", "val_accuracy"], ax=axs[1], grid=True)
    plt.savefig(f"{logging_dir}/train_val.svg", bbox_inches="tight")

    if verbose:
        plt.show()
    else:
        plt.close()


def train_single_imcnn(data_path,
                       n_epochs,
                       logging_dir=None,
                       adapt_data=None,
                       train_data=None,
                       val_data=None,
                       test_data=None,
                       skip_validation=False,
                       skip_testing=False):
    """Trains a single Intrinsic Mesh CNN on a subset of the PartNet dataset."""
    model = SegImcnn(
        adapt_data=PartNetDataset(data_path, set_type=0, only_signal=True) if adapt_data is None else adapt_data
    )
    train_hist = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }
    for epoch in range(n_epochs):
        # Reset generators for new epoch
        if train_data is not None:
            train_data.reset()
        if val_data is not None:
            val_data.reset()

        # Training
        epoch_train_hist = model.train_loop(
            dataset=PartNetDataset(set_type=0, path_to_zip=data_path) if train_data is None else train_data,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters()),
            verbose=True,
            epoch=epoch
        )
        train_hist["train_loss"].append(float(epoch_train_hist["epoch_loss"].detach()))
        train_hist["train_accuracy"].append(float(epoch_train_hist["epoch_accuracy"].detach()))

        # Validation
        if not skip_validation:
            epoch_val_hist = model.validation_loop(
                dataset=PartNetDataset(set_type=1, path_to_zip=data_path) if val_data is None else val_data,
                loss_fn=nn.CrossEntropyLoss(),
                verbose=True
            )
            train_hist["val_loss"].append(float(epoch_val_hist["val_epoch_loss"].detach()))
            train_hist["val_accuracy"].append(float(epoch_val_hist["val_epoch_accuracy"].detach()))

    # Testing
    if not skip_testing:
        epoch_test_hist = model.validation_loop(
            dataset=PartNetDataset(set_type=2, path_to_zip=data_path) if test_data is None else test_data,
            loss_fn=nn.CrossEntropyLoss(),
            verbose=False
        )
        train_hist["test_loss"].append(float(epoch_test_hist["val_epoch_loss"].detach()))
        train_hist["test_accuracy"].append(float(epoch_test_hist["val_epoch_accuracy"].detach()))

    if logging_dir is not None:
        log_training(model, train_hist, logging_dir)

    return model, train_hist
