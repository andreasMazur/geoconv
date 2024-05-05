from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import os
import torch


def log_summary(training_histories, n_epochs, logging_dir, verbose=False):
    """Computes mean and variances of training, validation and test statistics given for multiple runs."""
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


def log_training(model, hist, logging_dir, skip_validation, skip_testing, verbose=True):
    """Saves model, training history and plots training statistics."""
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Save model
    torch.save(model.state_dict(), f"{logging_dir}/model.zip")

    # Split train/val from test
    if skip_validation:
        train_val_df = pd.DataFrame.from_dict({
            "train_loss": hist["train_loss"],
            "train_accuracy": hist["train_accuracy"]
        })
    else:
        train_val_df = pd.DataFrame.from_dict({
            "train_loss": hist["train_loss"],
            "train_accuracy": hist["train_accuracy"],
            "val_loss": hist["val_loss"],
            "val_accuracy": hist["val_accuracy"],
        })
    if not skip_testing:
        test_df = pd.DataFrame.from_dict({
            "test_loss": hist["test_loss"],
            "test_accuracy": hist["test_accuracy"]
        })

    # Save csv files
    train_val_df.to_csv(f"{logging_dir}/train_val.csv", index=False)
    if not skip_testing:
        test_df.to_csv(f"{logging_dir}/test.csv", index=False)

    # Save plot
    n_rows, n_cols, cm = 2, 1, 1 / 2.54
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12 * cm, 12.2 * cm), sharex=True)
    axs[0].set_ylabel("Loss")
    if skip_validation:
        train_val_df.plot(y=["train_loss"], ax=axs[0], grid=True)
    else:
        train_val_df.plot(y=["train_loss", "val_loss"], ax=axs[0], grid=True)

    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epoch")
    if skip_validation:
        train_val_df.plot(y=["train_accuracy"], ax=axs[1], grid=True)
    else:
        train_val_df.plot(y=["train_accuracy", "val_accuracy"], ax=axs[1], grid=True)

    plt.savefig(f"{logging_dir}/train_val.svg", bbox_inches="tight")

    if verbose:
        plt.show()
    else:
        plt.close()