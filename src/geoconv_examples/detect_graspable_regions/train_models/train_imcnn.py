from geoconv_examples.detect_graspable_regions.data.dataset import PartNetDataset
from geoconv_examples.detect_graspable_regions.models.imcnn import SegImcnn

from torch import nn

import torch


def train_single_imcnn(data_path, saving_path, n_epochs):
    """Trains a single Intrinsic Mesh CNN on a subset of the PartNet dataset."""
    model = SegImcnn(adapt_data=PartNetDataset(data_path, set_type=0, only_signal=True))
    for epoch in range(n_epochs):
        model.train_loop(
            dataset=PartNetDataset(set_type=0, path_to_zip=data_path),
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters()),
            verbose=True,
            epoch=epoch
        )
        model.validation_loop(
            dataset=PartNetDataset(set_type=1, path_to_zip=data_path),
            loss_fn=nn.CrossEntropyLoss(),
            verbose=True
        )
    torch.save(model.state_dict(), saving_path)
