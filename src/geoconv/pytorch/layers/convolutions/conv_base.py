from geoconv.preprocessing.bc.bc_utils import create_template_matrix

from abc import abstractmethod
from torch import nn

import torch


class ConvBase(nn.Module):
    def __init__(self,
                 feature_input_dim,
                 n_radial,
                 n_angular,
                 template_radius,
                 include_kernel,
                 activation_fn,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Configure template
        self.template_radius = template_radius
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.template_vertices = torch.tensor(
            create_template_matrix(
                n_radial=int(self.n_radial),
                n_angular=int(self.n_angular),
                radius=float(self.template_radius),
                in_cart=False,
                exp_lambda=1.,
                shift_angular=False
            )
        )

        # Remember feature dimension
        self.feature_dim = feature_input_dim

        # Configure activation function
        self.activation_fn = activation_fn

        # Configure kernel
        self.include_kernel = include_kernel
        if self.include_kernel:
            self.kernel = torch.from_numpy(self.define_kernel_values(self.template_vertices.numpy())).to(torch.float32)
        else:
            self.kernel = None

    def forward(self, inputs):
        raise NotImplementedError

    def _gather_signals(self, barycentric_coordinates, mesh_signal, bc_with_angles=False):
        """Gathers required feature vectors and associates those to given barycentric coordinates.

        Parameters
        ----------
        barycentric_coordinates: torch.Tensor
            The barycentric coordinates tensor.
        mesh_signal: torch.Tensor
            The feature vectors at the mesh vertices.
        bc_with_angles: bool
            Whether the barycentric coordinates have angles concatenated to them.

        Returns
        -------
        (torch.Tensor, torch.Tensor):
            A tensor of shape (n_batch, n_vertices, n_radial, n_angular, 3, input_dim) that contains the required
            feature vectors and their according interpolation values in a tensor of shape
            (n_batch, n_vertices, n_radial, n_angular, 3).
        """
        # n_batch, n_vertices, n_radial, n_angular, 3, [2|3]
        bc_shape = barycentric_coordinates.size()

        # (n_batch, n_vertices * n_radial * n_angular * 3)
        if bc_with_angles:
            bc_values, bc_indices, _ = torch.unbind(barycentric_coordinates, dim=-1)
        else:
            bc_values, bc_indices = torch.unbind(barycentric_coordinates, dim=-1)
        bc_indices = bc_indices.reshape(bc_shape[0], -1).to(torch.int32)

        # (n_batch, n_vertices * n_radial * n_angular * 3, input_dim)
        batch_idx = torch.arange(bc_shape[0], device=bc_indices.device)[:, None]
        mesh_signal = mesh_signal[batch_idx, bc_indices]

        # (n_batch, n_vertices, n_radial, n_angular, 3, input_dim)
        mesh_signal = torch.reshape(
            mesh_signal, (bc_shape[0], bc_shape[1], bc_shape[2], bc_shape[3], 3, self.feature_dim)
        )
        return mesh_signal, bc_values

    def _patch_operator(self, mesh_signal, barycentric_coordinates):
        """Implementation of the patch-operator: weighting and signal-pullback.

        Parameters
        ----------
        mesh_signal: tf.Tensor
            The feature vectors that shall be interpolated at the template vertices.
        barycentric_coordinates: tf.Tensor
            The barycentric coordinates for the template vertices.

        Returns
        -------
        tf.Tensor:
            A tensor containing weighted (in case pre-defined kernels are used) and interpolated mesh signals of shape
            (batch_shapes, vertices, radial, angular, input_dim).
        """
        # interpolations : (batch_shapes, vertices, radial, angular, input_dim)
        interpolations = self._signal_pullback(mesh_signal, barycentric_coordinates)

        if self.include_kernel:
            # Weight matrix  : (radial, angular, radial, angular)
            # interpolations : (batch_shapes, vertices, radial, angular, input_dim)
            # Result         : (batch_shapes, vertices, radial, angular, input_dim)
            return torch.einsum("raxy,skxyf->skraf", self.kernel, interpolations)
        else:
            return interpolations

    def _signal_pullback(self, mesh_signal, barycentric_coordinates):
        """Implementation of signal-pullback: Weighting of gathered and interpolated mesh signals.

        Parameters
        ----------
        mesh_signal: tf.Tensor
            The signal values at the template vertices.
        barycentric_coordinates: tf.Tensor
            The barycentric coordinates for the template vertices.

        Returns
        -------
        tf.Tensor:
            A tensor containing interpolated feature vectors at the template vertices of shape
            (n_batch, n_vertices, n_radial, n_angular, input_dim).
        """

        # (n_batch, n_vertices, n_radial, n_angular, input_dim)
        return torch.sum(barycentric_coordinates.unsqueeze(dim=-1) * mesh_signal, dim=-2)

    def _signal_pullback_with_parallel_transport(self, signals, bc, rotation_order_vector):
        """Wrapper function for feature gathering, parallel transport and interpolation.

        Parameters
        ----------
        signals: torch.Tensor
            The surface signal.
            Shape: (batch, n_vertices, input_dim)
        bc: torch.Tensor
            The barycentric coordinates tensor with angles.
            Shape: (batch, n_vertices, n_radial, n_angular, 3, 3)
        rotation_order_vector: torch.Tensor
            A vector describing the rotation orders of the individual geometric components.
            Shape: (input_dim / 2)

        Returns
        -------
        torch.Tensor:
            Transported and interpolated feature vectors for each template vertex split into individual geometric
            components. The tensor has shape (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2).
        """
        # Get template vertex interpolations
        # neighbor_signals : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim)
        # bc_coefficients  : (n_batch, n_vertices, n_radial, n_angular, 3)
        neighbor_signals, bc_coefficients = self._gather_signals(bc, signals, bc_with_angles=True)

        # Reshape gathered signals into their geometric components
        # neighbor_signals : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2)
        signals_shape = neighbor_signals.size()
        n_geometric_components = rotation_order_vector.size(0)
        neighbor_signals = torch.reshape(
            neighbor_signals,
            (
                signals_shape[0],
                signals_shape[1],
                self.n_radial,
                self.n_angular,
                3,
                n_geometric_components,
                2
            )
        )

        # Prepare rotation matrices for parallel transport
        # rotation_matrices : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2, 2)
        rotation_matrices = self.create_rotation_matrices(bc, rotation_order_vector)

        # Transport via rotation and interpolate signals at template vertices
        # bc_coefficients   : (n_batch, n_vertices, n_radial, n_angular, 3)
        # rotation_matrices : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2, 2)
        # neighbor_signals  : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2)
        # result            : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        return torch.einsum(
            "bkral,bkralnxy,bkralny->bkranx", bc_coefficients, rotation_matrices, neighbor_signals
        )

    def create_rotation_matrices(self, bc, rotation_order):
        """Creates rotation matrices from given angles for parallel transport.

        Rotation matrix used:
            [cos ng, -sin ng]
            [sin ng, cos ng],
        whereby 'n = rotation_order'. If 'n = 0' rotation matrix becomes unit matrix
        and leaves features invariant.

        Parameters
        ----------
        bc: tf.Tensor
            The barycentric coordinates tensor.
        rotation_order: tf.Tensor
            The rotation order vector contains coefficients for how fast individual components of a
            feature vector rotate. It has shape (input_dim / 2).

        Returns
        -------
        tf.Tensor:
            Rotation matrices for the parallel transport.
            Shape: (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2, 2)
        """
        ### Gather the rotations angles ###
        # result     : (n_batch, n_vertices, n_radial, n_angular, 3)
        angles = bc[..., -1]

        ### Include rotation order ###
        # rotation_order : (      1,          1,        1,         1, 1, input_dim / 2)
        # angles         : (n_batch, n_vertices, n_radial, n_angular, 3,             1)
        # result         : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2)
        angles = rotation_order[None, None, None, None, None, :] * angles[..., None]

        ### Translate angles to rotation matrices ###
        # angles : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2)
        # result : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2, 2)
        C = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        S = torch.tensor([[0., -1.], [1., 0.]])
        rot_matrices = torch.cos(angles)[..., None, None] * C + torch.sin(angles)[..., None, None] * S

        # 'rot_matrices': (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2, 2)
        return rot_matrices

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
