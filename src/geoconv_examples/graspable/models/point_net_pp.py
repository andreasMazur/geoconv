from torch import nn
from torch_geometric.nn.conv import PointNetConv

import torch_geometric
import torch
import fpsample
import numpy as np


class MaxResponse(nn.Module):
    def forward(self, x):
        return x[torch.linalg.vector_norm(x, dim=-1).argmax()].view(1, -1)


class SetAbstraction(nn.Module):
    def __init__(self, n_balls, group_radius, local_nn_channel_list, global_nn_channel_list, max_amount_neighbors=None):
        super().__init__()
        self.n_balls = n_balls
        self.group_radius = group_radius
        # -1 due to centroid counting as a group member
        self.max_amount_neighbors = max_amount_neighbors

        self.point_net_conv = PointNetConv(
            local_nn=torch_geometric.nn.models.MLP(channel_list=local_nn_channel_list, act="relu"),
            global_nn=torch.nn.Sequential(
                MaxResponse(),  # Take max response from point embeddings for neighborhood
                torch_geometric.nn.models.MLP(channel_list=global_nn_channel_list, act="relu")
            ),
            add_self_loops=False  # Already handled by provided edge tensor
        )

    def forward(self, vertices):
        # 1. Farthest point sampling
        centroid_indices = fpsample.fps_sampling(np.array(vertices), self.n_balls)
        # 2. Grouping
        embeddings = []
        for neighborhood, centroid_idx, edges in self.group_around_centroids(centroid_indices, vertices):
            # 3. PointNet layer
            embeddings.append(self.point_net_conv(neighborhood, neighborhood, edges))
        # Return tensor of size: (self.n_balls, d + C')
        return torch.cat(embeddings, dim=0)

    def group_around_centroids(self, centroid_indices, vertices):
        # Compute vertex distances
        squared_norm = torch.einsum("ij,ij->i", vertices, vertices)
        vertex_distances = torch.sqrt(
            torch.abs(squared_norm - 2 * torch.einsum("ik,jk->ij", vertices, vertices) + squared_norm)
        )[centroid_indices]

        # Determine radii for groups while taking group limit into account.
        # I.e. if the amount of vertices exceeds the group limit, the group radius is reduced s.t.
        # group limit is maintained.
        if self.max_amount_neighbors is not None:
            group_distance_limits = vertex_distances.sort(dim=-1)[0][:, self.max_amount_neighbors]
            group_distance_limits[group_distance_limits > self.group_radius] = self.group_radius
        else:
            group_distance_limits = torch.tensor([self.group_radius]).repeat(vertex_distances.shape[0])

        # Filter vertices to groups
        vertices_in_range = vertex_distances <= group_distance_limits.view(-1, 1)
        for idx, neighborhood_mask in enumerate(vertices_in_range):
            # Determine neighborhoods indices
            vertices_in_neigh = torch.where(neighborhood_mask)[0]
            centroid_idx_in_neigh = torch.where(centroid_indices[idx] == vertices_in_neigh)[0]

            # Determine edges to neighborhood centroid
            n_neighbors = vertices_in_neigh.shape[0]
            edges_to_centroid = torch.stack(
                [torch.arange(n_neighbors), centroid_idx_in_neigh.repeat(n_neighbors)],
                dim=0
            )

            # Return neighborhood vertex coordinates and edges to centroid
            yield vertices[neighborhood_mask].float(), centroid_idx_in_neigh, edges_to_centroid
