from geoconv.tensorflow.layers import ConvIntrinsic

from typing import override

import tensorflow as tf
import numpy as np


def irrep_n_0(n, theta, option):
    """Computes the irreducible representation of type 'n' to 0 for angular coordinate 'theta'.

    Implements the \rho_n \to \rho_0 irrep for from:
    > [Gauge Equivariant Mesh CNNs: Anisotropic Convolutions on Geometric Graphs](https://arxiv.org/abs/2003.05425)
    > Pim de Haan, Maurice Weiler, Taco Cohen and Max Welling

    Parameters
    ----------
    n: int
        The input type of the irreducible representation.
    theta: float
        The angular coordinate of the irreducible representation.
    option: int
        There are two equally valid options for this irreducible representation. Choose either 0 or 1.

    Returns
    -------
    np.ndarray:
        The irreducible representation of type 'n' for angular coordinate 'theta'.
    """
    assert option in [0, 1], "For irrep n to 0 you only have options 0 or 1."
    if option == 0:
        a = np.cos(n * theta)
        b = np.sin(n * theta)
    else:
        a = np.sin(n * theta)
        b = -np.cos(n * theta)
    return np.array([[a, b]])


def irrep_0_m(m, theta, option):
    """Computes the irreducible representation of type 0 to 'm' for angular coordinate 'theta'.

    Implements the \rho_0 \to \rho_m irrep for from:
    > [Gauge Equivariant Mesh CNNs: Anisotropic Convolutions on Geometric Graphs](https://arxiv.org/abs/2003.05425)
    > Pim de Haan, Maurice Weiler, Taco Cohen and Max Welling

    Parameters
    ----------
    m: int
        The output type of the irreducible representation.
    theta: float
        The angular coordinate of the irreducible representation.
    option: int
        There are two equally valid options for this irreducible representation. Choose either 0 or 1.

    Returns
    -------
    np.ndarray:
        The irreducible representation of type 'n' for angular coordinate 'theta'.
    """
    assert option in [0, 1], "For irrep 0 to n you only have options 0 or 1."
    if option == 0:
        a = np.cos(m * theta)
        b = np.sin(m * theta)
    else:
        a = np.sin(m * theta)
        b = -np.cos(m * theta)
    return np.array([[a], [b]])


def irrep_n_to_m(n, m, theta, option):
    """Computes the irreducible representation of type 'n' to 'm' for angular coordinate 'theta'.

    Implements the \rho_n \to \rho_m irrep for from:
    > [Gauge Equivariant Mesh CNNs: Anisotropic Convolutions on Geometric Graphs](https://arxiv.org/abs/2003.05425)
    > Pim de Haan, Maurice Weiler, Taco Cohen and Max Welling

    Parameters
    ----------
    n: int
        The input type of the irreducible representation.
    m: int
        The output type of the irreducible representation.
    theta: float
        The angular coordinate of the irreducible representation.
    option: int
        There are two equally valid options for this irreducible representation. Choose either 0 or 1.

    Returns
    -------
    np.ndarray:
        The irreducible representation of type 'n' for angular coordinate 'theta'.
    """
    assert option in [0, 1, 2, 3], "For irrep 0 to n you only have options 0, 1, 2 or 3."
    if option == 0:
        a = np.cos((m - n) * theta)
        b = -np.sin((m - n) * theta)
        c = np.sin((m - n) * theta)
        d = np.cos((m - n) * theta)
    elif option == 1:
        a = np.sin((m - n) * theta)
        b = np.cos((m - n) * theta)
        c = -np.cos((m - n) * theta)
        d = np.sin((m - n) * theta)
    elif option == 2:
        a = np.cos((m + n) * theta)
        b = np.sin((m + n) * theta)
        c = np.sin((m + n) * theta)
        d = -np.cos((m + n) * theta)
    else:
        a = -np.sin((m + n) * theta)
        b = np.cos((m + n) * theta)
        c = np.cos((m + n) * theta)
        d = np.sin((m + n) * theta)
    return np.array([[a, b], [c, d]])


class ConvIntrinsicGE(ConvIntrinsic):
    def __init__(self, feature_types_in, feature_types_out, options, *args, **kwargs):
        """Initializes the gauge-equivariant surface convolution.

        Parameters
        ----------
        feature_types_in: list
            A list of integers representing the geometric type of the input features.
        feature_types_out: list
            A list of integers representing the geometric type of the output features.
        options: list
            A list of integers representing the selected input-output-dependent geometric types.
        """
        super().__init__(*args, **kwargs)
        self.feature_types_in = feature_types_in
        self.feature_types_out = feature_types_out

        assert len(options) == len(feature_types_in) * len(feature_types_out), \
            "You need to provide 'len(feature_types_in) * len(feature_types_out)' options."
        self.options = options

        # K is initialized in build
        self.K = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        # Additionally initialize K
        self.construct_group_repr()

    def construct_group_repr(self):
        """Constructs the group representation for the preset input-output feature types.

        Compare Section 3.1 in:
        > [Gauge Equivariant Mesh CNNs: Anisotropic Convolutions on Geometric Graphs](https://arxiv.org/abs/2003.05425)
        > Pim de Haan, Maurice Weiler, Taco Cohen and Max Welling
        """
        # We create a rotation matrix for each theta (in radians)
        K = []
        for theta in [((2 * k * np.pi) / self._all_rotations) * (np.pi / 180) for k in range(self._all_rotations)]:
            rows = []
            idx = 0
            # One row for each output type
            for f_out in self.feature_types_out:
                columns = []
                # One column for each input type
                for f_in in self.feature_types_in:
                    # Determine what irrep we need
                    if f_in == f_out == 0:
                        rot_matrix = np.array([[1]])
                    elif f_out == 0:
                        rot_matrix = irrep_n_0(n=f_in, theta=theta, option=self.options[idx])
                    elif f_in == 0:
                        rot_matrix = irrep_0_m(m=f_out, theta=theta, option=self.options[idx])
                    else:
                        rot_matrix = irrep_n_to_m(n=f_in, m=f_out, theta=theta, option=self.options[idx])
                    columns.append(rot_matrix)
                    idx += 1
                rows.append(np.concatenate(columns, axis=1))
            theta_matrix = np.concatenate(rows, axis=0)
            K.append(theta_matrix)
        self.K = np.stack(K)

    def call(self, inputs, **kwargs):
        mesh_signal, bary_coordinates, parallel_transport_angles = inputs

        ####################################################################
        # TODO: Fold center - conv_center: (batch_shapes, vertices, 1, templates)
        ####################################################################

        ###################################################################################
        # Fold neighbors - conv_neighbor: (batch_shapes, vertices, n_rotations, templates)
        ###################################################################################
        # Call patch operator
        interpolations = self._patch_operator(mesh_signal, bary_coordinates, parallel_transport_angles)

    @override
    def _patch_operator(self, mesh_signal, barycentric_coordinates, parallel_transport_angles):
        interpolations = self._signal_pullback(mesh_signal, barycentric_coordinates, parallel_transport_angles)

        if self.include_prior:
            # Weight matrix  : (radial, angular, radial, angular)
            # interpolations : (batch_shapes, vertices, radial, angular, input_dim)
            # Result         : (batch_shapes, vertices, radial, angular, input_dim)
            return tf.einsum("raxy,skxyf->skraf", self._kernel, interpolations)
        else:
            return interpolations

    @override
    def _signal_pullback(self, mesh_signal, barycentric_coordinates, parallel_transport_angles):
        # n_batch, n_vertices, n_radial, n_angular, 3, 2
        bc_shape = tf.shape(barycentric_coordinates)

        # (n_batch, n_vertices * n_radial * n_angular * 3)
        bc_indices, bc_values = tf.unstack(barycentric_coordinates, axis=-1)
        bc_indices = tf.cast(
            tf.reshape(bc_indices, (bc_shape[0], -1)), tf.int32
        )

        # (n_batch, n_vertices * n_radial * n_angular * 3, input_dim)
        mesh_signal = tf.gather(mesh_signal, bc_indices, batch_dims=1)

        # (n_batch, n_vertices, n_radial, n_angular, 3, input_dim)
        mesh_signal = tf.reshape(
            mesh_signal, (bc_shape[0], bc_shape[1], bc_shape[2], bc_shape[3], 3, self._feature_dim)
        )

        # Parallel transport via rotation
        # K: (n_angular, f_out, f_in) <- Can't use this because it would alter the mesh signal's dimension.
        # Here, we need the block-diagonal matrix dependent on the chosen geometric type:
        # [\rho_0(theta), 0, 0]
        # [0, \rho_1(theta), 0]
        # [0, 0, \rho_1(theta)]

        # (n_batch, n_vertices, n_radial, n_angular, input_dim)
        return tf.reduce_sum(tf.expand_dims(bc_values, axis=-1) * mesh_signal, axis=-2)

    def define_kernel_values(self, template_matrix):
        """Temporary for testing.
        TODO: Delete this later.
        """
        interpolation_coefficients = np.zeros(
            template_matrix.shape[:-1] + template_matrix.shape[:-1]
        )
        for mean_rho_idx in range(template_matrix.shape[0]):
            for mean_theta_idx in range(template_matrix.shape[1]):
                interpolation_coefficients[
                    mean_rho_idx, mean_theta_idx, mean_rho_idx, mean_theta_idx
                ] = 1.0
        return interpolation_coefficients


if __name__ == "__main__":
    layer = ConvIntrinsicGE(
        amt_templates=0,
        template_radius=1.0,
        feature_types_in=[0, 1, 1],  # Input dimension thus: 1 + 2 + 2 = 5
        feature_types_out=[1, 3],  # Input dimension thus: 2 + 2 = 4  => Thus creating 4 x 5 matrix K
        options=[0, 0, 0, 0, 0, 0],
        include_prior=False,
    )
    signal_shape = [6890, 5]
    barycentric_shape = [6890, 3, 4, 3, 2]
    layer.build([signal_shape, barycentric_shape])
