from geoconv_examples.detect_graspable_regions.partnet_grasp.dataset import PartNetGraspDataset
from geoconv_examples.detect_graspable_regions.training.imcnn import SegImcnn
from geoconv_examples.detect_graspable_regions.training.train_logging import log_training

from torch import nn

import torch


def train_single_imcnn(data_path,
                       n_epochs,
                       logging_dir=None,
                       adapt_data=None,
                       train_data=None,
                       val_data=None,
                       test_data=None,
                       skip_validation=False,
                       skip_testing=False,
                       verbose=False):
    """Trains a single Intrinsic Mesh CNN on a subset of the PartNet dataset."""
    model = SegImcnn(
        adapt_data=PartNetGraspDataset(data_path, set_type=0, only_signal=True) if adapt_data is None else adapt_data
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
            dataset=PartNetGraspDataset(set_type=0, path_to_zip=data_path) if train_data is None else train_data,
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
                dataset=PartNetGraspDataset(set_type=1, path_to_zip=data_path) if val_data is None else val_data,
                loss_fn=nn.CrossEntropyLoss(),
                verbose=True
            )
            train_hist["val_loss"].append(float(epoch_val_hist["val_epoch_loss"].detach()))
            train_hist["val_accuracy"].append(float(epoch_val_hist["val_epoch_accuracy"].detach()))

    # Testing
    if not skip_testing:
        epoch_test_hist = model.validation_loop(
            dataset=PartNetGraspDataset(set_type=2, path_to_zip=data_path) if test_data is None else test_data,
            loss_fn=nn.CrossEntropyLoss(),
            verbose=False
        )
        train_hist["test_loss"].append(float(epoch_test_hist["val_epoch_loss"].detach()))
        train_hist["test_accuracy"].append(float(epoch_test_hist["val_epoch_accuracy"].detach()))

    if logging_dir is not None:
        log_training(model, train_hist, logging_dir, skip_validation, skip_testing, verbose=verbose)

    return model, train_hist


def train_n_imcnns(n, data_path, n_epochs, logging_dir=None, verbose=False):
    """Train n IMCNNs."""
    training_histories = []
    for repetition in range(n):
        print(f"### Repetition {repetition} ###")
        _, train_hist = train_single_imcnn(
            data_path, n_epochs=n_epochs, logging_dir=f"{logging_dir}/rep_{repetition}", verbose=verbose
        )
        training_histories.append(train_hist)
    return training_histories
