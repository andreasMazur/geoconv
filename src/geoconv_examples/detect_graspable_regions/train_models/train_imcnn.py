from geoconv_examples.detect_graspable_regions.data.dataset import PartNetDataset
from geoconv_examples.detect_graspable_regions.models.imcnn import SegImcnn

from torch import nn

import torch


def train_single_imcnn(data_path,
                       saving_path,
                       n_epochs,
                       adapt_data=None,
                       train_data=None,
                       val_data=None,
                       test_data=None):
    """Trains a single Intrinsic Mesh CNN on a subset of the PartNet dataset."""
    model = SegImcnn(
        adapt_data=PartNetDataset(data_path, set_type=0, only_signal=True) if adapt_data is None else adapt_data
    )
    train_hist = dict()
    for epoch in range(n_epochs):
        # Training
        train_hist = model.train_loop(
            dataset=PartNetDataset(set_type=0, path_to_zip=data_path) if train_data is None else train_data,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters()),
            verbose=True,
            epoch=epoch
        )
        train_hist["train_loss"] = train_hist["epoch_loss"]
        train_hist["train_accuracy"] = train_hist["epoch_accuracy"]

        # Validation
        val_hist = model.validation_loop(
            dataset=PartNetDataset(set_type=1, path_to_zip=data_path) if test_data is None else test_data,
            loss_fn=nn.CrossEntropyLoss(),
            verbose=True
        )
        train_hist["val_loss"] = val_hist["val_epoch_loss"]
        train_hist["val_accuracy"] = val_hist["val_epoch_accuracy"]
    torch.save(model.state_dict(), saving_path)

    # Testing
    test_hist = model.validation_loop(
        dataset=PartNetDataset(set_type=2, path_to_zip=data_path) if val_data is None else val_data,
        loss_fn=nn.CrossEntropyLoss(),
        verbose=False
    )
    train_hist["test_loss"] = test_hist["val_epoch_loss"]
    train_hist["test_accuracy"] = test_hist["val_epoch_accuracy"]

    return test_hist
