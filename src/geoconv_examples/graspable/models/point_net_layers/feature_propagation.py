from torch import nn
from torch_geometric.nn.conv import PointNetConv

import torch_geometric
import torch


class FeaturePropagation(nn.Module):

    def __init__(self, channel_list, k=3, p=2, custom_point_net=None):
        super().__init__()
        self.k = k
        self.p = p

        if custom_point_net is None:
            # Emulate "unit-PointNet" by only using self loops, i.e. edge index is empty
            self.unit_point_net = PointNetConv(
                local_nn=torch_geometric.nn.models.MLP(channel_list=channel_list, act="relu"), add_self_loops=True
            )
        else:
            assert isinstance(custom_point_net, PointNetConv), "Please provide a PointNetConv network."
            self.unit_point_net = custom_point_net
        self.edge_index = torch.tensor([], dtype=torch.int32).view(2, 0)

    def inverse_distance_weighting(self, vertices, centroids, centroid_features):
        # 1.) Find k nearest neighbors (nc = nearest centroid)
        distances = torch.linalg.vector_norm(vertices.view(-1, 1, 3) - centroids, dim=-1)
        nc_distances, nc_indices = distances.topk(k=self.k, dim=-1, largest=False)

        # 2.) Compute inverse distance weights
        non_zero = (nc_distances != 0).all(dim=-1)
        inverse_weights = torch.zeros_like(nc_distances)
        inverse_weights[non_zero] = 1 / nc_distances[non_zero] ** self.p
        inverse_weights[nc_distances == 0] = 1.

        # 3.) Inverse distance weighting at 'vertices' using 'centroid_features'
        nc_features = centroid_features[nc_indices]
        interpolated_features = (
                (inverse_weights.view(-1, self.k, 1) * nc_features).sum(dim=1) / inverse_weights.sum(dim=-1).view(-1, 1)
        ).float()

        return interpolated_features

    def forward(self, vertices, centroids, centroid_features):
        interpolated_features = self.inverse_distance_weighting(vertices, centroids, centroid_features)
        # Use empty edge index to only use self loops
        return self.unit_point_net(interpolated_features, vertices.float(), self.edge_index)
