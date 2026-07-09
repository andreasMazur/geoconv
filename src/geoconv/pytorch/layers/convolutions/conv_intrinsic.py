from geoconv.pytorch.layers.convolutions.conv_base import ConvBase

from abc import abstractmethod
from torch import nn

import torch


class ConvIntrinsic(ConvBase):
    def __init__(self, output_dim, rotation_delta, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Remember layer attributes
        self.output_dim = int(output_dim)
        self.rotation_delta = int(rotation_delta)

        # Init neighbor weights
        self._template_neighbor_weights = nn.Parameter(
            torch.empty(self.output_dim, self.n_radial, self.n_angular, self.feature_dim)
        )
        nn.init.xavier_uniform_(self._template_neighbor_weights)

        # Init self weights
        self._template_self_weights = nn.Parameter(
            torch.empty(self.output_dim, 1, self.feature_dim)
        )
        nn.init.xavier_uniform_(self._template_self_weights)

        # Init bias
        self._bias = nn.Parameter(torch.empty(1, self.output_dim))
        nn.init.xavier_uniform_(self._bias)

    def forward(self, inputs):
        """Computes intrinsic surface convolution on all vertices of a given mesh.

        Parameters
        ----------
        **kwargs
        inputs: (torch.Tensor, torch.Tensor)
            The first tensor represents the signal defined on the manifold. It has size
            (batch_shapes, n_vertices, feature_dim). The second tensor represents the barycentric coordinates. It has
            size (batch_shapes, n_vertices, n_radial, n_angular, 3, 2).
        orientations: torch.Tensor
            Contains an integer that tells how to rotate the signal-interpolations.

        Returns
        -------
        torch.Tensor
            The geodesic convolution of the template with the signal on the object mesh in every given GPC-system.
            It has size (batch_shapes, vertices, n_rotations, templates)
        """
        mesh_signal, bary_coordinates = inputs

        ####################################################################
        # Fold center - conv_center: (batch_shapes, vertices, 1, templates)
        ####################################################################
        # Weight matrix : (templates, 1, input_dim)
        # Mesh signal   : (batch_shapes, vertices, input_dim)
        # Result        : (batch_shapes, vertices, 1, templates)
        conv_center = torch.einsum("tef,skf->sket", self._template_self_weights, mesh_signal)

        #####################################################################
        # Fold neighbors - conv_neighbor: (batch_shapes, vertices, n_rotations, templates)
        #####################################################################
        # Gather signals for patch operator
        # mesh_signal: (batch_shapes, vertices, radial, angular, 3, input_dim)
        # bc_values: (batch_shapes, vertices, radial, angular, 3)
        mesh_signal, bc_values = self._gather_signals(bary_coordinates, mesh_signal)

        # Call patch operator
        interpolations = self._patch_operator(mesh_signal, bc_values)

        def fold_neighbor(o):
            # Weight              : (templates, radial, angular, input_dim)
            # Mesh interpolations : (batch_shapes, vertices, radial, angular, input_dim)
            # Result              : (batch_shapes, vertices, templates)

            # torch.concat([interpolations[..., -1:, :], interpolations[..., :-1, :]], dim=-2)
            return torch.einsum(
                "traf,skraf->skt",
                self._template_neighbor_weights,
                torch.roll(interpolations, shifts=o, dims=-2),
            )

        # conv_neighbor: (batch_shapes, vertices, n_rotations, templates)
        orientations = torch.reshape(torch.arange(start=0, end=self.n_angular, step=self.rotation_delta), (-1, 1))
        conv_neighbor = torch.stack([fold_neighbor(o) for o in orientations])
        conv_neighbor = conv_neighbor.permute(1, 2, 0, 3)

        # result: (batch_shapes, vertices, n_rotations, templates)
        return self.activation_fn(conv_center + conv_neighbor + self._bias)

    @abstractmethod
    def define_kernel_values(self, template_matrix):
        """Defines the kernel values for each template vertex.

        Parameters
        ----------
        template_matrix: np.ndarray
            An array of size [n_radial, n_angular, 2], which contains the positions of the template vertices in
            polar coordinates.

        Returns
        -------
        np.ndarray:
            An array of size [n_radial, n_angular, n_radial, n_angular], which contains the interpolation weights for
            the patch operator '[D(x)f](rho_in, theta_in)'
        """
        pass
