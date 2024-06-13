from torch import nn

import torch


class AngularMaxPooling(nn.Module):
    """The implementation for angular max-pooling"""

    def forward(self, inputs):
        """Max-pools over the results of a intrinsic surface convolution.

        Parameters
        ----------
        inputs: torch.Tensor
            A tensor of size: (batch_shapes,  n_vertices, n_rotations, feature_dim), where 'n_vertices' references to
            the total amount of vertices in the triangle mesh, 'n_rotations' to the amount of rotations considered
            during the intrinsic surface convolution and 'feature_dim' to the feature dimensionality.

        Returns
        -------
        torch.Tensor:
            A three-dimensional tensor of size (batch_shapes, n_vertices, feature_dim).
        """
        return torch.max(inputs, dim=-2)[0]
