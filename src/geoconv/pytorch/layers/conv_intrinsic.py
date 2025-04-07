from geoconv.preprocessing.barycentric_coordinates import create_template_matrix

from abc import ABC, abstractmethod
from torch import nn
from typing import Optional, Tuple

import torch


ACTIVATIONS = {
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "selu": nn.SELU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "linear": nn.Identity(),
}

INITIALIZER = {
    "uniform": nn.init.uniform_,
    "normal": nn.init.normal_,
    "constant": nn.init.constant_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "trunc_normal": nn.init.trunc_normal_,
    "sparse_": nn.init.sparse_
}


def batched_fold_neighbor(template_neighbor_weights: torch.Tensor, interpolations: torch.Tensor, orientations: torch.Tensor) -> torch.Tensor:
    """Vectorized einsum across multiple orientations.

    Attributes:
    ----------
    template_neighbor_weights: torch.Tensor
        The template neighbor weights.
        Shape: (templates, radial, angular, input_dim)
    interpolations: torch.Tensor
        The interpolations.
        Shape: (batch_shapes, vertices, radial, angular, input_dim)
    orientations: torch.Tensor
        The orientations.
        Shape: (n_rotations)
    returns:
        torch.Tensor
        The stacked results of the einsum operations. (n_rotations, batch_shapes, vertices, templates)
    """
    rolled = []
    for orientation in orientations:
        rolled.append(
            torch.roll(
                interpolations,
                shifts=orientation.item(),
                dims=-2
            )
        )
    rotated_interpolations = torch.stack(rolled, dim=0).float()

    # Broadcast template_neighbor_weights to match the shape (n_rotations, templates, radial, angular, input_dim)
    expanded_weights = template_neighbor_weights.unsqueeze(0).expand(orientations.shape[0], -1, -1, -1, -1)

    # Apply vmap over orientations
    return torch.vmap(
        lambda weights, interps: torch.einsum("traf,skraf->skt", weights, interps)
    )(expanded_weights, rotated_interpolations)

class ConvIntrinsic(torch.jit.ScriptModule):
    """A metaclass for intrinsic surface convolutions on Riemannian manifolds.

    Attributes
    ----------
    activation_fn: str
        The activation function to use.
    rotation_delta: int
        The distance between two rotations. If `n` angular coordinates are given in the data, then the default behavior
        is to rotate the template `n` times, i.e. a shift in every angular coordinate of 1 to the next angular
        coordinate.
        If `rotation_delta = 2`, then the shift increases to 2 and the total amount of rotations reduces to
        ceil(n / rotation_delta). This gives a speed-up and saves memory. However, quality of results might worsen.
    amt_templates: int
        The amount of templates to apply during one convolution.
    template_radius: float
        The maximal geodesic extension of the template.
    initializer: str
        The initializer for the weights.
    include_prior: bool
        Whether to weight the interpolations according to a pre-defined kernel.
    """

    def __init__(self,
                 input_shape: Tuple[torch.Size, torch.Size],
                 amt_templates: int,
                 template_radius: float,
                 include_prior: bool = True,
                 activation: str = "relu",
                 rotation_delta: int = 1,
                 initializer: str = "xavier_uniform"):
        super().__init__()
        self.activation_fn = activation
        self.rotation_delta = rotation_delta
        self.amt_templates = amt_templates
        self.template_radius = template_radius
        self.initializer = initializer
        self.include_prior = include_prior

        # Attributes that depend on the data and are set automatically in build
        self._activation = ACTIVATIONS[self.activation_fn]
        self._init_fn = INITIALIZER[self.initializer]
        self._bias = None
        self._all_rotations = None
        self._template_size = None  # (#radial, #angular)
        self._template_vertices = None
        self._template_neighbor_weights = None
        self._template_self_weights = None
        self._feature_dim = None

        self.build(input_shape)

    def build(self, input_shape):
        """Builds the layer by setting template and bias attributes

        Parameters
        ----------
        input_shape: (torch.Size, torch.Size)
            The shape of the signal and the shape of the barycentric coordinates.
        """
        signal_shape, barycentric_shape = input_shape

        # signal_shape: (8, 784, 1), barycentric_shape: (8, 784, 3, 4, 3, 2)
        # Configure template
        self._template_size = (barycentric_shape[-4], barycentric_shape[-3])
        self._all_rotations = self._template_size[1]
        self._template_vertices = torch.from_numpy(
            create_template_matrix(self._template_size[0], self._template_size[1], radius=self.template_radius),
        )
        self._feature_dim = signal_shape[-1]

        # Configure trainable weights
        self._template_neighbor_weights = nn.Parameter(
            torch.zeros(size=(self.amt_templates, self._template_size[0], self._template_size[1], signal_shape[-1]))
        )
        self._init_fn(self._template_neighbor_weights)

        self._template_self_weights = nn.Parameter(torch.zeros(size=(self.amt_templates, 1, signal_shape[-1])))
        self._init_fn(self._template_self_weights)

        self._bias = nn.Parameter(torch.zeros(size=(self.amt_templates,1)))
        self._init_fn(self._bias)

        # Configure kernel
        self._configure_kernel()

    def forward(self, mesh_signal: torch.Tensor, bary_coordinates :torch.Tensor, orientations: Optional[torch.Tensor] = None):
        """Computes intrinsic surface convolution on all vertices of a given mesh.

        Parameters
        ----------
        inputs: (torch.Tensor, torch.Tensor)
        mesh_signal: (torch.Tensor)
            Represents the signal defined on the manifold.
            It has size (n_vertices, feature_dim).

        bary_coordinates: (torch.Tensor) 
            Represents the barycentric coordinates.
            It has size (n_vertices, n_radial, n_angular, 3, 2).
        orientations: torch.Tensor
            Contains an integer that tells how to rotate the data.

        Returns
        -------
        torch.Tensor
            The geodesic convolution of the template with the signal on the object mesh in every given GPC-system.
            It has size (vertices, n_rotations, templates)
        """
        ######################################################
        # Fold center - conv_center: (vertices, 1, templates)
        ######################################################
        # Weight matrix : (templates, 1, input_dim)
        # Mesh signal   : (batch_shapes, vertices, input_dim)
        # Result        : (batch_shapes, vertices, 1, n_templates)
        conv_center = torch.einsum("tef,skf->sket", self._template_self_weights, mesh_signal.to(torch.float32))

        #####################################################################
        # Fold neighbors - conv_neighbor: (vertices, n_rotations, templates)
        #####################################################################
        # Call patch operator
        interpolations = self._patch_operator(mesh_signal.to(torch.float32), bary_coordinates.to(torch.float32))
        # Determine orientations
        if orientations is None:
            # No specific orientations given. Hence, compute for all orientations.
            orientations = torch.arange(start=0, end=self._all_rotations, step=self.rotation_delta, device=interpolations.device)

        # Result: (batch_shapes, vertices, n_rotations, templates)
        conv_neighbor = torch.permute(
            batched_fold_neighbor(
                self._template_neighbor_weights,
                interpolations,
                orientations
            ),
            dims=[1, 2, 0, 3]
        )
        # conv_neighbor: (batch_shapes, vertices, n_rotations, templates)
        return self._activation(conv_center + conv_neighbor + self._bias.view(-1))

    def _patch_operator(self, mesh_signal, barycentric_coordinates):
        """Interpolates and weights mesh signal

        Parameters
        ----------
        mesh_signal: torch.Tensor
            The signal values at the template vertices
        barycentric_coordinates: torch.Tensor
            The barycentric coordinates for the template vertices

        Returns
        -------
        torch.Tensor:
            Weighted and interpolated mesh signals
        """
        interpolations = self._signal_pullback(mesh_signal.to(torch.float32), barycentric_coordinates.to(torch.float32))

        if self.include_prior:
            # Weight matrix  : (radial, angular, radial, angular)
            # interpolations : (batch_shapes, vertices, radial, angular, input_dim)
            # Result         : (batch_shapes, vertices, radial, angular, input_dim)
            return torch.einsum("raxy,skxyf->skraf", self._kernel, interpolations)
        else:
            return interpolations

    def _signal_pullback(self, mesh_signal, barycentric_coordinates):
        """Interpolates signals at template vertices

        Parameters
        ----------
        mesh_signal: torch.Tensor
            The signal values at the template vertices
        barycentric_coordinates: torch.Tensor
            The barycentric coordinates for the template vertices

        Returns
        -------
        torch.Tensor:
            Interpolation values for the template vertices
        """
        mesh_signal.to(torch.float32)
        barycentric_coordinates.to(torch.float32)

        # Get relevant shapes
        B, n_v, F = mesh_signal.shape
        v_idx, n_r, n_a, K = barycentric_coordinates.shape[1:5]

        # Split indices and weights 
        vertex_indices = barycentric_coordinates[..., 0].long()
        weights = barycentric_coordinates[..., 1]

        # Expand mesh signal and indices so we can use torch.gather to gather along the vertex dimension
        mesh_signal_exp = mesh_signal.view(B, n_v, 1, 1, 1, F).expand(B, n_v, n_r, n_a, K, F)
        indices_exp = vertex_indices.unsqueeze(-1).expand(B, v_idx, n_r, n_a, K, F)

        gathered = torch.gather(mesh_signal_exp, dim=1, index=indices_exp)

        # Weigh gathered
        interpolated = (gathered * weights.unsqueeze(-1))

        # Return aggregated interpolations
        return interpolated.sum(dim=-2)


    def _configure_kernel(self):
        """Defines all necessary interpolation coefficient matrices for the patch operator."""
        self.register_buffer('_kernel', self.define_kernel_values(self._template_vertices).to(torch.float32))

    @abstractmethod
    def define_kernel_values(self, template_matrix):
        """Defines the kernel values for each template vertex.

        Parameters
        ----------
        template_matrix: torch.Tensor
            An array of size [n_radial, n_angular, 2], which contains the positions of the template vertices in
            cartesian coordinates.

        Returns
        -------
        torch.Tensor:
            An array of size [n_radial, n_angular, n_radial, n_angular], which contains the interpolation weights for
            the patch operator '[D(x)f](rho_in, theta_in)'
        """
        pass
