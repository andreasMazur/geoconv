from geoconv_examples.detect_graspable_regions.data.dataset import PartNetDataset
from geoconv_examples.detect_graspable_regions.train_models.train_imcnn import log_training

from torch import nn

import torch


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
        log_training(model, train_hist, logging_dir)

    return model, train_hist
