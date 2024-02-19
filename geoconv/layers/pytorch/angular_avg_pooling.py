from torch import nn

import torch


class AngularAvgPooling(nn.Module):
    """The implementation for angular max-pooling"""

    def forward(self, inputs):
        """Max-pools over the results of a geodesic convolution.

        Parameters
        ----------
        inputs: tf.Tensor
            The result tensor of an intrinsic surface convolution.
            It has a size of: (n_vertices, n_rotations, feature_dim), where 'n_vertices' references to the total amount
            of vertices in the triangle mesh, 'n_rotations' to the amount of rotations considered during the intrinsic
            surface convolution and 'feature_dim' to the feature dimensionality.

        Returns
        -------
        tf.Tensor:
            A two-dimensional tensor of size (n_vertices, feature_dim), that contains a convolution results for each
            vertex. Thereby, the convolution result has the largest Euclidean norm among the convolution results for
            all rotations.
        """
        return torch.mean(inputs, dim=1)
