from geoconv.pytorch.utils.compute_shot_lrf import compute_neighborhood
from torch import nn

import torch


class EuclNeighborsDescriptor(nn.Module):
    def __init__(self, n_neighbors, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors

    def forward(self, inputs):
        # 'neighborhoods' : (batch, vertices, n_neighbors, 3)
        neighborhoods, _, _ = compute_neighborhood(inputs, self.n_neighbors)

        # Scale max sphere radius to 1
        max_norm = torch.amax(torch.linalg.norm(neighborhoods, axis=-1))
        neighborhoods = torch.where(max_norm != 0., neighborhoods / max_norm, 0.)[..., None, None]
        neighborhoods_shape = neighborhoods.size()
        return torch.reshape(
            neighborhoods, (neighborhoods_shape[0], neighborhoods_shape[1], self.n_neighbors * 3)
        )[..., 3:]  # Cut away the origin-zero vectors
