from geoconv.pytorch.layers.convolutions.conv_base import ConvBase

from torch import nn

import torch


class ConvHarmonic(ConvBase):
    def __init__(self, output_dim, rotation_order, *args, **kwargs):
        """Initializes the object.

        Parameters
        ----------
        output_dim: torch.Tensor
            The output dim.
        rotation_order: int
            The rotation order.
        *args: tuple
            The args.
        **kwargs: dict
            The kwargs.
        """
        kwargs.pop("include_kernel", None)
        super().__init__(include_kernel=False, *args, **kwargs)

        # Check that input- and output vectors are divisible by 2
        assert self.feature_dim % 2 == 0, "This layer requires the dimension of input features to be divisible by two."
        assert output_dim % 2 == 0, "This layer requires the dimension of output features to be divisible by two."

        # Define output dimension
        self.output_dim = int(output_dim)

        # Remember rotation order
        self.rotation_order = int(rotation_order)

        # Compute how many complex numbers the input vectors contain
        self.n_complex_num_output = int(self.output_dim // 2)

        # Build layer
        self.n_complex_num_input = int(self.feature_dim // 2)
        self.rotation_order_vector = torch.full(
            size=(self.n_complex_num_input,), fill_value=self.rotation_order
        ).to(torch.float32)

        self._radial_weights = nn.Parameter(
            torch.empty(self.n_radial, self.n_complex_num_output, self.n_complex_num_input)
        )
        nn.init.xavier_uniform_(self._radial_weights)

        self._radial_weights_center = nn.Parameter(torch.empty(self.n_complex_num_output, self.n_complex_num_input))
        nn.init.xavier_uniform_(self._radial_weights_center)

        self._phase_offset = nn.Parameter(torch.empty(1, self.n_complex_num_output))
        nn.init.xavier_uniform_(self._phase_offset)

        self._all_angular_coordinates = self.template_vertices[0, :, 1].to(torch.float32)

    def create_phase_weight_tensor(self):
        """Creates phase weight tensor Phi.

        Returns
        -------
        torch.Tensor:
            The phase weight matrix computed with the current weights.
            Shape: (n_angular, output_dim, 2, 2)
        """
        # Compute arguments (i.e., angles) for trigonometric functions
        # Rotation_order           : ()
        # _all_angular_coordinates : (n_angular,)
        # _phase_offset            : (output_dim / 2, 1)
        # angles                   : (n_angular, output_dim / 2)
        angles = (self.rotation_order * self._all_angular_coordinates)[:, None] + self._phase_offset

        # Apply trigonometric functions
        # cos_matrix: (n_angular, output_dim / 2)
        # sin_matrix: (n_angular, output_dim / 2)
        cos_matrix = torch.cos(angles)
        sin_matrix = torch.sin(angles)

        # Create final weight matrix tensor of shape (n_angular, output_dim / 2, 2, 2)
        I = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        K = torch.tensor([[0., -1.], [1., 0.]])
        return I * cos_matrix[..., None, None] + K * sin_matrix[..., None, None]

    def create_phase_weight_tensor_center(self):
        """Creates phase weight tensor Phi for center vertices. I.e., theta = 0.

        Returns
        -------
        torch.Tensor:
            The phase weight matrix computed with the current weights.
            Shape: (output_dim / 2, 2, 2)
        """
        # Apply trigonometric functions on phase offset
        # cos_matrix: (1, output_dim / 2,)
        # sin_matrix: (1, output_dim / 2,)
        cos_matrix = torch.cos(self._phase_offset)
        sin_matrix = torch.sin(self._phase_offset)

        # Create final weight matrix tensor of shape (1, output_dim / 2, 2, 2)
        I = torch.tensor([[1., 0.], [0., 1.]])
        K = torch.tensor([[0., -1.], [1., 0.]])
        return I * cos_matrix[..., None, None] + K * sin_matrix[..., None, None]

    def forward(self, inputs):
        """Computes the harmonic surface convolution.

        Parameters
        ----------
        inputs: (torch.Tensor, torch.Tensor)
            The first tensor has shape [n_batch, n_vertices, input_dim] and contains the signals for each vertex. The
            second tensor has shape [n_batch, n_vertices, n_radial, n_angular, 3, 2] and contains the barycentric
            coordinates.

        Returns
        -------
        torch:Tensor
            A tensor of size [n_batch, n_vertices, output_dim], containing the new signal-embeddings for each mesh
            vertex.
        """
        # signals : (n_batch, n_vertices, input_dim)
        # bc      : (n_batch, n_vertices, n_radial, n_angular, 3, 3)
        signals, bc = inputs

        # Get transported and interpolated feature vectors at each template vertex
        # neighbor_signals : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        neighbor_signals = self._signal_pullback_with_parallel_transport(signals, bc, self.rotation_order_vector)

        # Get phase weight tensor
        # phase_weights: (n_angular, output_dim / 2, 2, 2)
        phase_weights_neigh = self.create_phase_weight_tensor()

        # Aggregate neighbors
        # radial_weights      : (n_radial, output_dim / 2, input_dim / 2)
        # phase_weights_neigh : (n_angular, output_dim / 2, 2, 2)
        # neighbor_signals    : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        # conv_neigh          : (n_batch, n_vertices, output_dim / 2, 2)
        conv_neigh = torch.einsum(
            "rqf,aqxy,bkrafy->bkqx", self._radial_weights, phase_weights_neigh, neighbor_signals
        )

        # Reshape vertex signals into their geometric components
        # signals : (n_batch, n_vertices, input_dim / 2, 2)
        signals_shape = signals.size()
        signals = torch.reshape(signals, (signals_shape[0], signals_shape[1], self.n_complex_num_input, 2))

        # Compute self connections
        # radial_weights_center : (output_dim / 2, input_dim / 2)
        # phase_weights_center  : (1, output_dim / 2, 2, 2)
        # signals               : (n_batch, n_vertices, input_dim / 2, 2)
        # conv_center           : (n_batch, n_vertices, output_dim / 2, 2)
        phase_weights_center = self.create_phase_weight_tensor_center()
        conv_center = torch.einsum(
            "qf,eqxy,bkfy->bkqx", self._radial_weights_center, phase_weights_center, signals
        )

        # Add self-connection contributions to neighbor aggregation for complete conv result
        # conv_center : (n_batch, n_vertices, output_dim / 2, 2)
        # conv_neigh  : (n_batch, n_vertices, output_dim / 2, 2)
        # result      : (n_batch, n_vertices, output_dim / 2, 2)
        result = conv_center + conv_neigh

        # Apply magnitude activation
        # result_amp : (n_batch, n_vertices, output_dim / 2)
        # result     : (n_batch, n_vertices, output_dim / 2, 2)
        result_amp = torch.maximum(torch.linalg.norm(result, axis=-1), torch.tensor(1e-6))
        result = self.activation_fn(result_amp)[..., None] * torch.where(
            result_amp[..., None] != 0., result / result_amp[..., None], 0.
        )

        # Return output in original shape
        # (n_batch, n_vertices, output_dim)
        return torch.reshape(result, (signals_shape[0], signals_shape[1], self.output_dim))

    def define_kernel_values(self, template_matrix):
        """"HSNs do not use weighting functions to interpolate features among template vertices."""
        return None
