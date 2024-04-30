from torch import nn
from torch_geometric.nn.conv import PointNetConv

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

        self.point_net_conv = PointNetConv()

    def forward(self, vertex_features, vertices):
        # 1. Farthest point sampling
        centroid_indices = fpsample.fps_sampling(np.array(vertices), self.subsample_size)

        # 2. Grouping
        for neighborhoods, edges in self.group_around_centroids(centroid_indices, vertices):
            # 3. PointNet
            # neighborhood_embeddings = self.point_net_conv()
            pass

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
        vertices_in_range = vertex_distances <= group_distance_limits.view((-1, 1))
        amount_neighbors = torch.sum(vertices_in_range, dim=-1)

        # Generate groups
        for n_neighbors in torch.unique(amount_neighbors):
            # Get neighborhoods with n neighbors
            neighborhood_group_mask = amount_neighbors == n_neighbors

            # Get neighbor indices for selected neighborhoods
            centroid_neighbor_masks = vertices_in_range[neighborhood_group_mask]
            centroid_neighbor_indices = torch.where(centroid_neighbor_masks)[1].view((-1, n_neighbors))

            # Translate global indices to local indices
            neighborhood_centroids = torch.tensor(
                centroid_indices[neighborhood_group_mask], dtype=torch.int32
            )
            new_centroid_indices = torch.where(neighborhood_centroids.view((-1, 1)) == centroid_neighbor_indices)[1]

            # Get edges from neighbors towards centroids
            new_neighbor_indices = torch.arange(n_neighbors)
            centroid_edges = torch.stack(
                [
                    new_neighbor_indices.view(1, -1).repeat(torch.sum(neighborhood_group_mask), 1),
                    new_centroid_indices.view(-1, 1).repeat(1, n_neighbors)
                ],
                dim=1
            )

            # Yield neighborhoods with same amount of neighbors and edge indices
            # Group coordinates: (#groups, #neighbors, 3)
            yield vertices[centroid_neighbor_indices], centroid_edges


if __name__ == "__main__":
    sa_layer = SetAbstraction(subsample_size=3445, group_radius=0.2, group_limit=512)

    mesh = trimesh.load_mesh("/home/andreas/Uni/datasets/MPI-FAUST/training/registrations/tr_reg_000.ply")
    sa_layer(torch.tensor(mesh.vertices), torch.tensor(mesh.vertices))
