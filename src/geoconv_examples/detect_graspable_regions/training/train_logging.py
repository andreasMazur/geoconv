from matplotlib import pyplot as plt

import pandas as pd
import os
import torch


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
