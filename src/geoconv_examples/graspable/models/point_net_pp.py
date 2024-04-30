from torch import nn
from torch_geometric.nn.conv import PointNetConv

import torch_geometric
import torch
import trimesh
import fpsample
import numpy as np


class SetAbstraction(nn.Module):
    def __init__(self, subsample_size, group_radius, group_limit):
        super().__init__()
        self.subsample_size = subsample_size
        self.group_radius = group_radius
        # -1 due to centroid counting as a group member
        self.group_limit = group_limit - 1

        self.point_net_conv = PointNetConv(
            local_nn=torch_geometric.nn.models.MLP(channel_list=[6, 64, 64, 128], act="relu")
        )

    def forward(self, vertices):
        # 1. Farthest point sampling
        centroid_indices = fpsample.fps_sampling(np.array(vertices), self.subsample_size)

        # 2. Grouping
        for neighborhood, edges in self.group_around_centroids(centroid_indices, vertices):

            # 3. PointNet layer
            neighborhood_embeddings = self.point_net_conv(neighborhood, neighborhood, edges)

    def group_around_centroids(self, centroid_indices, vertices):
        # Compute vertex distances
        squared_norm = torch.einsum("ij,ij->i", vertices, vertices)
        vertex_distances = torch.sqrt(
            torch.abs(squared_norm - 2 * torch.einsum("ik,jk->ij", vertices, vertices) + squared_norm)
        )[centroid_indices]

        # Determine radii for groups while taking group limit into account.
        # I.e. if the amount of vertices exceeds the group limit, the group radius is reduced s.t.
        # group limit is maintained.
        group_distance_limits = vertex_distances.sort(dim=-1)[0][:, self.group_limit]
        group_distance_limits[group_distance_limits > self.group_radius] = self.group_radius

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
            yield vertices[neighborhood_mask].float(), edges_to_centroid


if __name__ == "__main__":
    sa_layer = SetAbstraction(subsample_size=3445, group_radius=0.2, group_limit=512)

    mesh = trimesh.load_mesh("/home/andreas/Uni/datasets/MPI-FAUST/training/registrations/tr_reg_000.ply")
    sa_layer(torch.tensor(mesh.vertices))
