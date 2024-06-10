from torch import nn

import torch


class AngularMinPooling(nn.Module):
    """The implementation for angular max-pooling"""

    def forward(self, inputs):
        """Min-pools over the results of a intrinsic surface convolution.

        Parameters
        ----------
        inputs: torch.Tensor
            The result tensor of an intrinsic surface convolution.
            It has a size of: (n_vertices, n_rotations, feature_dim), where 'n_vertices' references to the total amount
            of vertices in the triangle mesh, 'n_rotations' to the amount of rotations considered during the intrinsic
            surface convolution and 'feature_dim' to the feature dimensionality.

        Returns
        -------
        torch.Tensor:
            A two-dimensional tensor of size (n_vertices, feature_dim), that contains a convolution results for each
            vertex. Thereby, the convolution result has the smallest Euclidean norm among the convolution results for
            all rotations.
        """
        minimal_response = torch.linalg.vector_norm(inputs, ord=2, dim=-1)
        minimal_response = torch.argmin(minimal_response, dim=-1).int()
        pooled_signals = []
        for signal, indices in zip(inputs, minimal_response):
            pooled_signals.append(signal[torch.arange(0, inputs.shape[1]), indices])
        return torch.stack(pooled_signals)
