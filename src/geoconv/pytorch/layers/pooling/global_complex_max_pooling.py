from torch import nn

import torch


class GlobalComplexPooling(nn.Module):
    def forward(self, inputs):
        """Takes the maximum value over each channel.

        Parameters
        ----------
        inputs: torch.Tensor
            A 'b x n x i' tensor, whereby 'b' represents the number of shapes, 'n' the number of vertices per
            shape and 'i' the input feature dimension.

        Returns
        -------
        torch.Tensor
            A 'b x i' tensor, where the i-th channel contains the complex number with the largest norm across all 'n'
            vertices.
        """
        # inputs_shape contains the following values: (batch, vertices, input_dim)
        inputs_shape = tuple(inputs.size())

        # Reshape inputs into complex value structure
        # inputs : (batch, vertices, input_dim / 2, 2)
        inputs = torch.reshape(inputs, inputs_shape[:-1] + (-1, 2))

        # Determine amplitudes
        # amplitudes : (batch, vertices, input_dim / 2)
        amplitudes = torch.linalg.norm(inputs, dim=-1)

        # Determine largest amplitudes among all vertices in each channel
        # index_largest_amplitude : (batch, input_dim / 2)
        index_largest_amplitude = torch.argmax(amplitudes, dim=-2)

        # Gather complex values with the largest amplitudes
        n_complex = int(inputs_shape[-1] / 2)

        # 'index_largest_amplitude': (batch, input_dim / 2, 3)
        index_largest_amplitude = torch.stack(
            [
                torch.arange(inputs_shape[0])[:, None].repeat(1, n_complex),  # indices for batch dimension
                index_largest_amplitude,  # indices for vertex dimension
                torch.arange(n_complex)[None, :].repeat(inputs_shape[0], 1)  # indices for channel dimension
            ],
            dim=-1
        )

        # inputs (batch, input_dim / 2, 2)
        inputs = inputs[index_largest_amplitude.unbind(-1)]

        # Reshape to original shape
        # inputs : (batch, input_dim)
        return torch.reshape(inputs, (inputs_shape[0], inputs_shape[-1]))
