from tensorflow.keras.layers import Layer, Activation

import tensorflow as tf


class ConvGeodesic(Layer):
    """The Tensorflow implementation of the geodesic convolution

    Paper, that introduced the geodesic convolution:
    > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://arxiv.org/abs/1501.06297)
    > Jonathan Masci and Davide Boscaini et al.

    Attributes
    ----------
    output_dim:
        The dimensionality of the output feature vectors.
    amt_kernel:
        The amount of kernels to apply during one convolution.
    activation:
        The activation function to use.
    rotation_delta:
        The distance between two rotations. If `n` angular coordinates are given in the data, then the default behavior
        is to rotate the kernel `n` times, i.e. a shift in every angular coordinate of 1 to the next angular coordinate.
        If `rotation_delta = 2`, then the shift increases to 2 and the total amount of rotations reduces to
        ceil(n / rotation_delta). This gives a speed up and saves memory. However, quality of results might worsen.
    """

    def __init__(self, output_dim, amt_kernel, activation="relu", name=None, rotation_delta=1):
        if name:
            super().__init__(name=name)
        else:
            super().__init__()

        self.activation = Activation(activation)
        self.output_dim = output_dim
        self.rotation_delta = rotation_delta
        self.amt_kernel = amt_kernel

        # Attributes that depend on the data and are set automatically in build
        self._kernel_size = None  # (#radial, #angular)
        self._kernels = None
        self._bias = None
        self._all_rotations = None

    def get_config(self):
        config = super(ConvGeodesic, self).get_config()
        config.update(
            {
                "kernel_size": self._kernel_size,
                "output_dim": self.output_dim,
                "activation": self.activation,
                "all_rotations": self._all_rotations,
                "rotation_delta": self.rotation_delta,
                "amt_kernel": self.amt_kernel
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
        self._kernel_size = (barycentric_shape[2], barycentric_shape[3])
        self._all_rotations = self._kernel_size[1]
        self._kernels = self.add_weight(
            name="GeoConvKernel",
            shape=(self.amt_kernel, self._kernel_size[0], self._kernel_size[1], self.output_dim, signal_shape[2]),
            initializer="glorot_uniform",
            trainable=True
        )
        self._bias = self.add_weight(
            name="GeoConvBias",
            shape=(self._kernel_size[0], self._kernel_size[1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True
        )

    @tf.function
    def call(self, inputs):
        """Computes geodesic convolutions for multiple given GPC-systems

        Parameters
        ----------
        inputs: (tf.Tensor, tf.Tensor)
            The first tensor represents a batch of signals defined on the manifold. It has size
            (n_batch, n_vertices, feature_dim). The second tensor represents a batch of barycentric coordinates. It has
            size (n_batch, n_gpc_systems, n_radial, n_angular, 3, 2).

        Returns
        -------
        tf.Tensor
            The geodesic convolution of the kernel with the signal on the object mesh in every given GPC-system for
            every rotation. It has size (n_batch, n_rotations, n_gpc_systems, self.output_dim)
        """

        signal, b_coordinates = inputs
        result_tensor = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        batch_size = tf.shape(signal)[0]
        for idx in tf.range(batch_size):
            new_signal = self._geodesic_convolution(signal[idx], b_coordinates[idx])
            result_tensor = result_tensor.write(idx, new_signal)
        return result_tensor.stack()

    @tf.function
    def _geodesic_convolution(self, signal, barycentric_coords):

        # Interpolate signals at kernel vertices
        interpolation_fn = lambda bc: self._interpolate(signal, bc)
        interpolation_values = tf.vectorized_map(interpolation_fn, barycentric_coords)
        interpolation_values = tf.expand_dims(interpolation_values, axis=1)

        # Compute rotations
        all_rotations_fn = lambda rot: tf.roll(interpolation_values, shift=rot, axis=3)
        # n_rotations = ceil(self.all_rotations / self.rotation_delta)
        interpolation_values = tf.vectorized_map(
            all_rotations_fn, tf.range(start=0, limit=self._all_rotations, delta=self.rotation_delta)
        )

        # Compute convolution
        # Shape kernel: (                            n_kernel, n_radial, n_angular, new_dim, feature_dim)
        # Shape values: (n_rotations, n_gpc_systems,        1, n_radial, n_angular,          feature_dim)
        # Shape result: (n_rotations, n_gpc_systems, n_kernel, n_radial, n_angular,          new_dim    )
        result = tf.linalg.matvec(self._kernels, interpolation_values)
        result = result + self._bias

        # Sum over all kernels (2), radial (3) and angular (4) coordinates
        # Shape result: (n_rotations, n_gpc_systems, new_dim)
        return tf.reduce_sum(result, axis=[2, 3, 4])

    @tf.function
    def _interpolate(self, signal, bary_coords):
        """Signal interpolation at kernel vertices

        This procedure was suggested in:
        > [Multi-directional Geodesic Neural Networks via Equivariant Convolution](https://arxiv.org/abs/1810.02303)
        > Adrien Poulenard and Maks Ovsjanikov

        Parameters
        ----------
        signal: tf.Tensor
            A tensor containing the signal defined on the entire object mesh. It has size (n_vertices, feature_dim).
        bary_coords: tf.Tensor
            A tensor containing the barycentric coordinates for one GPC-system. It has size (n_radial, n_angular, 3, 2).

        Returns
        -------
        tf.Tensor
            A tensor containing the interpolation values for all kernel vertices in the given GPC-system. It has size
            (n_radial, n_angular, feature_dim)
        """

        # Reshape for broadcasting convenience
        bary_coords = tf.reshape(bary_coords, (-1, 3, 2))

        # Gather vertex signals and weight them with Barycentric coordinates
        vertex_signals = tf.gather(signal, tf.cast(bary_coords[:, :, 0], tf.int32))
        vertex_signals = tf.multiply(vertex_signals, tf.expand_dims(bary_coords[:, :, 1], axis=-1))

        # Compute interpolation and reshape back to original shape
        interpolations = tf.math.reduce_sum(vertex_signals, axis=1)
        interpolations = tf.reshape(interpolations, (self._kernel_size[0], self._kernel_size[1], -1))
        return interpolations
