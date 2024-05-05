from geoconv_examples.detect_graspable_regions.data.dataset import PartNetDataset
from geoconv_examples.detect_graspable_regions.models.point_cnn import PointCNN
from geoconv_examples.detect_graspable_regions.models.point_net_pp import PointNetPP
from geoconv_examples.detect_graspable_regions.train_models.train_imcnn import log_training

from torch import nn
from matplotlib import pyplot as plt

import torch
import numpy as np
import pandas as pd


def train_single_model(model,
                       data_path,
                       n_epochs,
                       logging_dir=None,
                       train_data=None,
                       val_data=None,
                       test_data=None,
                       skip_validation=False,
                       skip_testing=False):
    """Trains a single Intrinsic Mesh CNN on a subset of the PartNet dataset."""
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
            opt=torch.optim.Adam(model.parameters()),
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
        log_training(model, train_hist, logging_dir, skip_validation, skip_testing)

    return model, train_hist


def train_n_models(model_variant, n, data_path, n_epochs, amount_classes=2, logging_dir=None, verbose=True):
    """Train the given model n times."""
    # Define possible models to train
    possible_models = {
        "PointCNN": PointCNN(amount_classes=amount_classes), "PointNetPP": PointNetPP(amount_classes=amount_classes)
    }

    # Train models
    training_histories = []
    for repetition in range(n):
        print(f"### Repetition {repetition} ###")
        model = possible_models[model_variant]
        _, train_hist = train_single_model(
            model, data_path, n_epochs=n_epochs, logging_dir=f"{logging_dir}/rep_{repetition}"
        )
        training_histories.append(train_hist)

    ###########
    # PLOTTING
    ###########
    # Compute epoch mean
    mean_train_loss = np.zeros((n_epochs,))
    mean_train_accuracy = np.zeros((n_epochs,))
    mean_val_loss = np.zeros((n_epochs,))
    mean_val_accuracy = np.zeros((n_epochs,))
    mean_test_loss = 0
    mean_test_accuracy = 0
    for train_hist in training_histories:
        mean_train_loss += np.array(train_hist["train_loss"])
        mean_train_accuracy += np.array(train_hist["train_accuracy"])
        mean_val_loss += np.array(train_hist["val_loss"])
        mean_val_accuracy += np.array(train_hist["val_accuracy"])
        mean_test_loss += train_hist["test_loss"][-1]
        mean_test_accuracy += train_hist["test_accuracy"][-1]
    mean_train_loss = mean_train_loss / n_epochs
    mean_train_accuracy = mean_train_accuracy / n_epochs
    mean_val_loss = mean_val_loss / n_epochs
    mean_val_accuracy = mean_val_accuracy / n_epochs
    mean_test_loss = mean_test_loss / n_epochs
    mean_test_accuracy = mean_test_accuracy / n_epochs

    # Compute epoch variance
    var_train_loss = np.zeros((n_epochs,))
    var_train_accuracy = np.zeros((n_epochs,))
    var_val_loss = np.zeros((n_epochs,))
    var_val_accuracy = np.zeros((n_epochs,))
    var_test_loss = 0
    var_test_accuracy = 0
    for train_hist in training_histories:
        var_train_loss = np.array(train_hist["train_loss"]) - mean_train_loss
        var_train_accuracy = np.array(train_hist["train_accuracy"]) - mean_train_accuracy
        var_val_loss = np.array(train_hist["val_loss"]) - mean_val_loss
        var_val_accuracy = np.array(train_hist["val_accuracy"]) - mean_val_accuracy
        var_test_loss = train_hist["test_loss"][-1] - mean_test_loss
        var_test_accuracy = train_hist["test_accuracy"][-1] - mean_test_accuracy
    var_train_loss = var_train_loss / (n_epochs - 1)
    var_train_accuracy = var_train_accuracy / (n_epochs - 1)
    var_val_loss = var_val_loss / (n_epochs - 1)
    var_val_accuracy = var_val_accuracy / (n_epochs - 1)
    var_test_loss = var_test_loss / (n_epochs - 1)
    var_test_accuracy = var_test_accuracy / (n_epochs - 1)

    # Save summary
    summary_hist_train_val = pd.DataFrame.from_dict({
        "mean_train_loss": mean_train_loss,
        "var_train_loss": var_train_loss,
        "mean_train_accuracy": mean_train_accuracy,
        "var_train_accuracy": var_train_accuracy,
        "mean_val_loss": mean_val_loss,
        "var_val_loss": var_val_loss,
        "mean_val_accuracy": mean_val_accuracy,
        "var_val_accuracy": var_val_accuracy,
    })
    summary_hist_train_val.to_csv(f"{logging_dir}/summary_train_val.csv", index=False)

    summary_hist_test = pd.DataFrame.from_dict({
        "mean_test_loss": [mean_test_loss],
        "var_test_loss": [var_test_loss],
        "mean_test_accuracy": [mean_test_accuracy],
        "var_test_accuracy": [var_test_accuracy]
    })
    summary_hist_test.to_csv(f"{logging_dir}/summary_test.csv", index=False)

    # Plot summary
    n_rows, n_cols, cm = 2, 1, 1 / 2.54
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12 * cm, 12.2 * cm), sharex=True)
    axs[0].set_ylabel("Loss")
    summary_hist_train_val.plot(
        y=["mean_train_loss", "mean_val_loss"],
        # yerr=["var_train_loss", "var_val_loss"],
        ax=axs[0],
        grid=True
    )
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epoch")
    summary_hist_train_val.plot(
        y=["mean_train_accuracy", "mean_val_accuracy"],
        # yerr=["var_train_accuracy", "var_val_accuracy"],
        ax=axs[1],
        grid=True
    )
    plt.savefig(f"{logging_dir}/summary_test.svg", bbox_inches="tight")
    if verbose:
        plt.show()
    else:
        plt.close()

    return training_histories
