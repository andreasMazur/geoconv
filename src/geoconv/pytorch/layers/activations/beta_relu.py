from torch import nn

import torch


class BetaRelu(nn.Module):
    def __init__(self, feature_input_dim, min_norm=1e-6, *args, **kwargs):
        """Initializes the object.

        Parameters
        ----------
        feature_input_dim: torch.Tensor
            The feature input dim.
        min_norm: float
            The min norm.
        *args: tuple
            The args.
        **kwargs: dict
            The kwargs.
        """
        super().__init__(*args, **kwargs)
        self.min_norm = min_norm
        self.beta = nn.Parameter(torch.empty(feature_input_dim // 2, 1))

    def forward(self, inputs):
        """Applies the layer to the inputs.

        Parameters
        ----------
        inputs: torch.Tensor
            The inputs tensor.

        Returns
        -------
        torch.Tensor
            A tensor containing the beta-relu altered inputs.
        """
        input_shape = inputs.size()
        inputs = torch.reshape(inputs, (input_shape[0], input_shape[1], input_shape[2] // 2, 2))
        inputs_norm = torch.maximum(torch.linalg.norm(inputs, axis=-1), torch.tensor(self.min_norm))
        inputs = torch.relu(inputs_norm - self.beta)[..., None] * torch.where(
            inputs_norm[..., None] != 0., inputs / inputs_norm[..., None], 0.
        )
        return torch.reshape(inputs, (input_shape[0], input_shape[1], input_shape[2]))
