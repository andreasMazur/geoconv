from src.geoconv_examples.graspable.models.point_net_layers.set_abstraction import SetAbstraction
from src.geoconv_examples.graspable.models.point_net_layers.feature_propagation import FeaturePropagation

from torcheval.metrics.functional import multiclass_accuracy
from torch import nn

import torch
import numpy as np
import sys


class PointNetPP(nn.Module):
    def __init__(self, amount_classes):
        """Initializes PointNet++ for segmentation

        # Taken from PointNet++ paper in the supplementary material (B.1)
        # ``Network for semantic and part segmentation''

        Attributes
        ----------
        amount_classes: int
            The amount of classes to segment.
        """
        super().__init__()
        coordinate_dim = 3

        ####################################
        # Initialize set-abstraction layers
        ####################################
        self.set_abstraction_parameters = [
            {
                "n_balls": 512,
                "group_radius": 0.2,
                "local_nn_channel_list": [3 + coordinate_dim, 64, 64],
                "global_nn_channel_list": [64, 128]
            },
            {
                "n_balls": 128,
                "group_radius": 0.4,
                "local_nn_channel_list": [128 + coordinate_dim, 128, 128],
                "global_nn_channel_list": [128, 256]
            },
            {  # Global set abstraction
                "n_balls": 1,
                "group_radius": np.inf,
                "local_nn_channel_list": [256 + coordinate_dim, 256, 512],
                "global_nn_channel_list": [512, 1024]
            }
        ]

        self.sa_layers = nn.ModuleList()
        for idx, conf in enumerate(self.set_abstraction_parameters):
            self.sa_layers.append(
                SetAbstraction(
                    n_balls=conf["n_balls"],
                    group_radius=conf["group_radius"],
                    local_nn_channel_list=conf["local_nn_channel_list"],
                    global_nn_channel_list=conf["global_nn_channel_list"]
                )
            )

        ########################################
        # Initialize feature propagation layers
        ########################################
        self.feature_prop_parameters = [
            {"channel_list": [1024 + coordinate_dim, 256, 256], "k": 1},
            {"channel_list": [256 + coordinate_dim, 128], "k": 3},
            {"channel_list": [128 + coordinate_dim, 128, 128, 128, amount_classes], "k": 3}
        ]

        self.fp_layers = nn.ModuleList()
        for idx, conf in enumerate(self.feature_prop_parameters):
            self.fp_layers.append(
                FeaturePropagation(channel_list=conf["channel_list"], k=conf["k"])
            )

        # TODO: Add dropout in between last two FC layers

    def forward(self, inputs):
        """Forwards through the model.

        Parameters
        ----------
        inputs: Tuple[torch.Tensor, torch.Tensor]
            The input to the model consists of vertex features and their associated vertices.

        Returns
        -------
        torch.Tensor
            New vertex features.
        """
        vertex_features, vertices = inputs

        # PointNet++ convolutions
        old_vertices = []
        for idx in range(len(self.set_abstraction_parameters)):
            old_vertices.append(vertices.clone())  # 0. 1. 2.
            vertex_features, vertices = self.sa_layers[idx](vertex_features, vertices)

        # Feature propagation
        for idx in range(len(self.feature_prop_parameters)):
            vertex_features = self.fp_layers[idx](old_vertices[-(idx + 1)], vertices, vertex_features)
            vertices = old_vertices[-(idx + 1)]

        return vertex_features

    def train_loop(self, dataset, loss_fn, opt, verbose=True, epoch=None):
        """Training loop."""
        # Go into training mode
        self.train()

        # Initialize statistics
        epoch_accuracy = 0.
        epoch_loss = 0.
        mean_accuracy = 0.
        mean_loss = 0.

        for step, ((vertices, _), gt) in enumerate(dataset):
            pred = self([vertices, vertices])
            loss = loss_fn(pred, gt)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Statistics
            epoch_accuracy = epoch_accuracy + multiclass_accuracy(pred, gt).detach()
            epoch_loss = epoch_loss + loss.detach()

            # I/O
            mean_accuracy = epoch_accuracy / (step + 1)
            mean_loss = epoch_loss / (step + 1)
            if verbose:
                sys.stdout.write(
                    f"\rEpoch: {epoch} - "
                    f"Training step: {step} - "
                    f"Loss {mean_loss:.4f} - "
                    f"Accuracy {mean_accuracy:.4f}"
                )

        return {"epoch_loss": mean_loss, "epoch_accuracy": mean_accuracy}
