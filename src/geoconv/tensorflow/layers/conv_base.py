from geoconv.preprocessing.bc.bc_utils import create_template_matrix

from abc import abstractmethod

import tensorflow as tf
import numpy as np


class ConvBase(tf.keras.layers.Layer):
    """A metaclass for intrinsic surface convolutions."""
    def __init__(self, n_radial, n_angular, template_radius, include_kernel, activation, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        # Configure template
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.template_radius = template_radius
        self.template_vertices = tf.constant(
            create_template_matrix(
                self.n_radial,
                self.n_angular,
                radius=self.template_radius,
                in_cart=False,
                exp_lambda=1.,
                shift_angular=False
            )
        )

        # Configure activation function
        self.activation = activation
        self.activation_fn = tf.keras.layers.Activation(self.activation)

        # Configure kernel
        self.include_kernel = include_kernel
        if self.include_kernel:
            self._kernel = tf.cast(
                self.define_kernel_values(self.template_vertices.numpy()), tf.float32
            )

        # Set in build the moment inputs have been seen
        self.feature_dim = None

    def build(self, inputs):
        signal_shape, barycentric_coordinates_shape = inputs
        self.feature_dim = signal_shape[-1]
        self.n_radial = barycentric_coordinates_shape[-4]
        self.n_angular = barycentric_coordinates_shape[-3]

    @tf.function
    def _patch_operator(self, mesh_signal, barycentric_coordinates):
        """Interpolates and weights mesh signal

        Parameters
        ----------
        mesh_signal: tf.Tensor
            The signal values at the template vertices.
        barycentric_coordinates: tf.Tensor
            The barycentric coordinates for the template vertices.

        Returns
        -------
        tf.Tensor:
            Weighted and interpolated mesh signals.
        """
        # interpolations : (batch_shapes, vertices, radial, angular, input_dim)
        interpolations = self._signal_pullback(mesh_signal, barycentric_coordinates)

        if self.include_kernel:
            # Weight matrix  : (radial, angular, radial, angular)
            # interpolations : (batch_shapes, vertices, radial, angular, input_dim)
            # Result         : (batch_shapes, vertices, radial, angular, input_dim)
            return tf.einsum("raxy,skxyf->skraf", self._kernel, interpolations)
        else:
            return interpolations

    @tf.function
    def _signal_pullback(self, mesh_signal, barycentric_coordinates):
        """Interpolates signals at template vertices

        Parameters
        ----------
        mesh_signal: tf.Tensor
            The signal values at the template vertices.
        barycentric_coordinates: tf.Tensor
            The barycentric coordinates for the template vertices.

        Returns
        -------
        tf.Tensor:
            Interpolation values for the template vertices.
        """

        # (n_batch, n_vertices, n_radial, n_angular, input_dim)
        return tf.reduce_sum(tf.expand_dims(barycentric_coordinates, axis=-1) * mesh_signal, axis=-2)

    @tf.function
    def _gather_signals(self, barycentric_coordinates, mesh_signal):
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
            mesh_signal, (bc_shape[0], bc_shape[1], bc_shape[2], bc_shape[3], 3, self.feature_dim)
        )
        return mesh_signal, bc_values

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
