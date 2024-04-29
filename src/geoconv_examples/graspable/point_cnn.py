from torch import nn
from torch_geometric.nn.conv import XConv
from torcheval.metrics.functional import multiclass_accuracy

import torch
import sys


class PointCNN(nn.Module):
    def __init__(self, amount_classes):
        """Initializes PointCNN for segmentation

        # Taken from PointCNN paper supplementary material Figure 1 (e)

        Attributes
        ----------
        amount_classes: int
            The amount of classes to segment.
        """
        super().__init__()

        # https://github.com/daerduoCarey/partnet_seg_exps/blob/master/exps/sem_seg_pointcnn/partnet_sem_seg.py:
        # "change sample_num 2,048 to 10,000, batch_size from 16 to 4, while keeping the other settings unchanged"
        sample_num = 2_048
        self.xconv_parameters = [
            # First "In" assumes 3D coordinates (as in PartNet paper assumed)
            {"N": sample_num, "In": 3, "C": 256, "K": 8, "D": 1, "hidden": 3},  # xconv
            {"N": 768, "In": 256, "C": 256, "K": 12, "D": 2, "hidden": 4},
            {"N": 384, "In": 256, "C": 512, "K": 16, "D": 2, "hidden": 4},
            {"N": 128, "In": 512, "C": 1024, "K": 16, "D": 6, "hidden": 4},
            {"N": 384, "In": 1024, "C": 512, "K": 16, "D": 6, "hidden": 4},  # xdconv
            {"N": 768, "In": 512, "C": 256, "K": 12, "D": 6, "hidden": 4},
            {"N": sample_num, "In": 256, "C": 256, "K": 8, "D": 6, "hidden": 4},
            {"N": sample_num, "In": 256, "C": 256, "K": 8, "D": 4, "hidden": 4}
        ]
        self.xconv_layers = nn.ModuleList()
        for idx, conf in enumerate(self.xconv_parameters):
            self.xconv_layers.append(
                XConv(
                    in_channels=conf["In"],
                    out_channels=conf["C"],
                    dim=3,  # Point cloud dimensionality
                    kernel_size=conf["K"],
                    hidden_channels=conf["hidden"],
                    dilation=conf["D"]
                )
            )

        # Weight sharing per vertex
        self.fc_parameters = [
            {"In": 256, "Out": 256},
            {"In": 256, "Out": 256},
            {"In": 256, "Out": amount_classes},
        ]
        self.fc_layers = nn.ModuleList()
        for conf in self.fc_parameters:
            # Missing activation?
            self.fc_layers.append(nn.Linear(in_features=conf["In"], out_features=conf["Out"]))

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

        # X-Convolutions
        skip_connections = []
        for idx in range(len(self.xconv_parameters)):
            if idx == 4:
                vertex_features = self.xconv_layers[idx](vertex_features, vertices)
                vertex_features += skip_connections[2]
            elif idx == 5:
                vertex_features = self.xconv_layers[idx](vertex_features, vertices)
                vertex_features += skip_connections[1]
            elif idx in [6, 7]:
                vertex_features = self.xconv_layers[idx](vertex_features, vertices)
                vertex_features += skip_connections[0]
            else:
                vertex_features = self.xconv_layers[idx](vertex_features, vertices)
                skip_connections.append(vertex_features.clone())

        # Linear layers
        for idx in range(len(self.fc_parameters)):
            vertex_features = self.fc_layers[idx](vertex_features)

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
