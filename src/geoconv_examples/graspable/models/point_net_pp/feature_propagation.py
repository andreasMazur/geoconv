from torch import nn
from torch_geometric.nn.conv import PointNetConv

import torch_geometric
import torch
import numpy as np


class FeaturePropagation(nn.Module):

    def __init__(self, channel_list, k=3, p=2):
        super().__init__()
        self.k = k
        self.p = p

        # Emulate "unit-PointNet" by only using self loops, i.e. edge index is empty
        self.unit_point_net = PointNetConv(
            local_nn=torch_geometric.nn.models.MLP(channel_list=channel_list, act="relu"), add_self_loops=True
        )
        self.edge_index = torch.tensor([], dtype=torch.int32).view(2, 0)

    def inverse_distance_weighting(self, vertices, centroids, centroid_features):
        # 1.) Find k nearest neighbors (nc = nearest centroid)
        distances = torch.linalg.vector_norm(vertices.view(-1, 1, 3) - centroids, dim=-1)
        nc_distances, nc_indices = distances.topk(k=self.k, dim=-1, largest=False)

        # 2.) Inverse distance weighting at 'vertices' using 'centroid_features'
        nc_features = centroid_features[nc_indices]
        inverse_weights = 1 / nc_distances ** self.p
        interpolated_features = (
                (inverse_weights.view(-1, self.k, 1) * nc_features).sum(dim=1) / inverse_weights.sum(dim=-1).view(-1, 1)
        ).float()

        # 3.) If infinite weights have been observed, replace interpolated with original centroid feature
        inf_weight_positions = inverse_weights == np.inf
        inf_nc = nc_indices[torch.where(inf_weight_positions)]
        interpolated_features[inf_weight_positions.any(dim=-1)] = centroid_features[inf_nc]

        return interpolated_features

    def forward(self, vertices, centroids, centroid_features):
        interpolated_features = self.inverse_distance_weighting(vertices, centroids, centroid_features)
        # Use empty edge index to only use self loops
        return self.unit_point_net(interpolated_features, vertices.float(), self.edge_index)
