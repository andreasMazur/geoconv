from geoconv.preprocessing.barycentric_coordinates import create_template_matrix

from tensorflow import keras
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np


class ConvIntrinsicLite(ABC, keras.layers.Layer):
    """A metaclass for intrinsic surface convolutions on Riemannian manifolds with smaller weight tensors.

    Attributes
    ----------
    output_dim:
        The dimensionality of the output vectors.
    amt_kernel:
        The amount of kernels to apply during one convolution.
    activation_fn:
        The activation function to use.
    rotation_delta:
        The distance between two rotations. If `n` angular coordinates are given in the data, then the default behavior
        is to rotate the kernel `n` times, i.e. a shift in every angular coordinate of 1 to the next angular coordinate.
        If `rotation_delta = 2`, then the shift increases to 2 and the total amount of rotations reduces to
        ceil(n / rotation_delta). This gives a speed-up and saves memory. However, quality of results might worsen.
    splits:
        The 'splits'-parameter determines into how many chunks the mesh signal is split. Each chunk will be folded
        iteratively to save memory. That is, fewer splits allow a faster convolution. More splits allow reduced memory
        usage. Careful: 'splits' has to divide the amount of vertices in the input mesh. Also, using many splits might
        cause larger memory fragmentation.
    """

    def __init__(self,
                 output_dim,
                 amt_kernel,
                 kernel_radius,
                 activation="relu",
                 rotation_delta=1,
                 splits=1,
                 name=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 initializer="glorot_uniform"):
        if name:
            super().__init__(name=name)
        else:
            super().__init__()

        self.activation_fn = activation
        self.output_dim = output_dim
        self.rotation_delta = rotation_delta
        self.amt_kernel = amt_kernel
        self.kernel_radius = kernel_radius
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.initializer = initializer
        self.splits = splits

        # Attributes that depend on the data and are set automatically in build
        self._activation = keras.layers.Activation(self.activation_fn)
        self._bias = None
        self._all_rotations = None
        self._kernel_size = None  # (#radial, #angular)
        self._kernel_vertices = None
        self._kernel_weights = None
        self._interpolation_coefficients = None
        self._feature_dim = None

    def get_config(self):
        config = super(ConvIntrinsicLite, self).get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "amt_kernel": self.amt_kernel,
                "kernel_radius": self.kernel_radius,
                "activation_fn": self.activation_fn,
                "rotation_delta": self.rotation_delta,
                "splits": self.splits,
                "name": self.name,
                "kernel_regularizer": self.kernel_regularizer,
                "bias_regularizer": self.bias_regularizer,
                "initializer": self.initializer
            }
        )
        return config

    def build(self, input_shape):
        """Builds the layer by setting kernel and bias attributes

        Parameters
        ----------
        input_shape: (tf.TensorShape, tf.TensorShape)
            The shape of the signal and the shape of the barycentric coordinates.
        """
        signal_shape, barycentric_shape = input_shape

        # Configure kernel
        self._kernel_size = (barycentric_shape[1], barycentric_shape[2])
        self._all_rotations = self._kernel_size[1]
        self._kernel_vertices = tf.constant(
            create_template_matrix(self._kernel_size[0], self._kernel_size[1], radius=self.kernel_radius)
        )
        self._feature_dim = signal_shape[-1]

        # Configure trainable weights
        self._kernel_weights = self.add_weight(
            name="conv_intrinsic_kernel",
            shape=(self.amt_kernel, self.output_dim, signal_shape[1]),
            initializer=self.initializer,
            trainable=True,
            regularizer=self.kernel_regularizer
        )
        self._bias = self.add_weight(
            name="conv_intrinsic_bias",
            shape=(self.amt_kernel, self.output_dim),
            initializer=self.initializer,
            trainable=True,
            regularizer=self.bias_regularizer
        )

        # Configure patch operator
        self._configure_patch_operator()

    @tf.function
    def call(self, inputs):
        """Computes intrinsic surface convolution for multiple given GPC-systems

        Parameters
        ----------
        inputs: (tf.Tensor, tf.Tensor)
            The first tensor represents the signal defined on the manifold. It has size
            (n_vertices, feature_dim). The second tensor represents the barycentric coordinates. It has
            size (n_vertices, n_radial, n_angular, 3, 2).

        Returns
        -------
        tf.Tensor
            The geodesic convolution of the kernel with the signal on the object mesh in every given GPC-system for
            every rotation. It has size (n_batch, n_rotations, n_vertices, feature_dim)
        """
        mesh_signal, bary_coordinates = inputs
        mesh_signal = self._patch_operator(mesh_signal, bary_coordinates)

        idx = tf.constant(0)
        new_signal = tf.TensorArray(
            tf.float32,
            size=self.splits,
            dynamic_size=False,
            clear_after_read=True,
            tensor_array_name="outer_ta",
            name="call_ta"
        )
        for interpolations in tf.split(mesh_signal, self.splits):
            # Compute convolution
            new_signal = new_signal.write(idx, self._fold(interpolations))
            idx = idx + tf.constant(1)
        new_signal = new_signal.concat()

        # Shape result: (n_vertices, n_rotations, self.output_dim)
        return new_signal

    @tf.function
    def _patch_operator(self, mesh_signal, barycentric_coordinates):
        """Implements the patch operator

        Parameters
        ----------
        mesh_signal: tf.Tensor
            The signal values at the kernel vertices
        barycentric_coordinates: tf.Tensor
            The barycentric coordinates for the kernel vertices

        Returns
        -------
        tf.Tensor:
            Interpolation values for the kernel vertices
        """
        ##########################################
        # Signal-interpolation at kernel vertices
        ##########################################
        mesh_signal = tf.reshape(
            tf.gather(mesh_signal, tf.reshape(tf.cast(barycentric_coordinates[:, :, :, :, 0], tf.int32), (-1,))),
            (-1, self._kernel_size[0], self._kernel_size[1], 3, self._feature_dim)
        )
        # (n_vertices, n_radial, n_angular, input_dim)
        mesh_signal = tf.math.reduce_sum(
            tf.expand_dims(barycentric_coordinates[:, :, :, :, 1], axis=-1) * mesh_signal, axis=-2
        )

        ##################
        # Including prior
        ##################
        # (subset, n_radial * n_angular, input_dim)
        mesh_signal = tf.reshape(mesh_signal, (-1, self._kernel_size[0] * self._kernel_size[1], self._feature_dim))

        # (subset, input_dim, n_radial * n_angular)
        mesh_signal = tf.transpose(mesh_signal, perm=[0, 2, 1])

        # (subset, 1, 1, input_dim, n_radial * n_angular)
        mesh_signal = tf.expand_dims(tf.expand_dims(mesh_signal, axis=1), axis=1)

        # (subset,       1,          1, input_dim, n_radial * n_angular)
        # (        n_radial, n_angular,            n_radial * n_angular)
        # (subset, n_radial, n_angular, input_dim                      )
        return tf.linalg.matvec(mesh_signal, self._interpolation_coefficients)

    @tf.function
    def _fold(self, interpolations):
        """Folds kernel vertex signal with the kernel weights

        Parameters
        ----------
        interpolations: tf.Tensor
            The according to a given weighting function weighted interpolations at the kernel vertices.
            Shape: (subset, n_radial, n_angular, input_dim)

        Returns
        -------
        tf.Tensor:
            The new mesh signal after the convolution. Shape: (subset, n_rotations, self.output_dim)
        """
        idx = tf.constant(0)
        all_rotations = tf.range(start=0, limit=self._all_rotations, delta=self.rotation_delta)
        size = tf.shape(all_rotations)[0]
        new_signal = tf.TensorArray(
            tf.float32,
            size=size,
            dynamic_size=False,
            clear_after_read=True,
            tensor_array_name="inner_ta",
            name="rotation_ta"
        )
        # Iterate over rotations to economize memory usage
        for rot in all_rotations:
            # Rotate
            # (subset, n_radial, n_angular, input_dim)
            rotated_interpolations = tf.roll(interpolations, shift=rot, axis=2)
            # (subset, n_radial * n_angular, input_dim)
            rotated_interpolations = tf.reshape(
                rotated_interpolations, (-1, self._kernel_size[0] * self._kernel_size[1], self._feature_dim)
            )
            # (subset, input_dim, n_radial * n_angular)
            rotated_interpolations = tf.transpose(rotated_interpolations, perm=[0, 2, 1])
            # (subset, 1, input_dim, n_radial * n_angular)
            rotated_interpolations = tf.expand_dims(rotated_interpolations, axis=1)
            # Matrix x Signal
            # Shape kernel : (        n_kernel, self.output_dim,            input_dim)
            # Shape input  : (subset,        1,       input_dim, n_radial * n_angular)
            # Shape result : (subset, n_kernel, self.output_dim, n_radial * n_angular)
            rotated_interpolations = self._kernel_weights @ rotated_interpolations
            # Sum over all (new) kernel vertex signals and add bias as well as activation
            # (subset, n_kernel, self.output_dim)
            rotated_interpolations = self._activation(tf.reduce_sum(rotated_interpolations, axis=-1) + self._bias)
            # Sum over kernel
            # (subset, self.output_dim)
            rotated_interpolations = tf.reduce_sum(rotated_interpolations, axis=1)
            # Output dim: (subset, self.output_dim)
            new_signal = new_signal.write(idx, rotated_interpolations)
            idx = idx + tf.constant(1)

        # New signal: (n_rotations, subset, self.output_dim)
        new_signal = new_signal.stack()
        # New signal: (subset, n_rotations, self.output_dim)
        return tf.transpose(new_signal, perm=[1, 0, 2])

    def _configure_patch_operator(self):
        """Defines all necessary interpolation coefficient matrices for the patch operator."""

        self._interpolation_coefficients = tf.cast(
            self.define_interpolation_coefficients(self._kernel_vertices.numpy()), tf.float32
        )
        self._interpolation_coefficients = tf.reshape(
            self._interpolation_coefficients,
            (self._kernel_size[0], self._kernel_size[1], self._kernel_size[0] * self._kernel_size[1])
        )

    @abstractmethod
    def define_interpolation_coefficients(self, kernel_matrix):
        """Defines the interpolation coefficients for each kernel vertex.

        Parameters
        ----------
        kernel_matrix: np.ndarray
            An array of size [n_radial, n_angular, 2], which contains the positions of the kernel vertices in cartesian
            coordinates.

        Returns
        -------
        np.ndarray:
            An array of size [n_radial, n_angular, n_radial, n_angular], which contains the interpolation weights for
            the patch operator '[D(x)f](rho_in, theta_in)'
        """
        pass
