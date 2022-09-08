from tensorflow.keras.layers import Layer, Activation

import tensorflow as tf


class ConvGeodesic(Layer):

    def __init__(self, output_dim, amt_kernel, activation="relu", name=None):
        if name:
            super().__init__(name=name)
        else:
            super().__init__()

        # Define kernel attributes
        self.kernels = []
        self.kernel_size = None  # (#radial, #angular)
        self.bias = None

        # Define output attributes
        self.output_dim = output_dim
        self.activation = Activation(activation)

        # Define convolution attributes
        self.all_rotations = None
        self.amt_kernel = amt_kernel

    def get_config(self):
        config = super(ConvGeodesic, self).get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "output_dim": self.output_dim,
                "activation": self.activation,
                "all_rotations": self.all_rotations,
                "amt_kernel": self.amt_kernel
            }
        )
        return config

    def build(self, input_shape):
        """

        """
        signal_shape, barycentric_shape = input_shape
        self.kernel_size = (barycentric_shape[2], barycentric_shape[3])
        self.all_rotations = self.kernel_size[1]
        self.kernels = self.add_weight(
            name="GeoConvKernel",
            shape=(self.amt_kernel, self.kernel_size[0], self.kernel_size[1], self.output_dim, signal_shape[2]),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="GeoConvBias",
            shape=(self.kernel_size[0], self.kernel_size[1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True
        )

    @tf.function
    def call(self, inputs):
        """

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
        """

        """

        # Interpolate signals at kernel vertices
        interpolation_fn = lambda bc: self._interpolate(signal, bc)
        interpolation_values = tf.vectorized_map(interpolation_fn, barycentric_coords)
        interpolation_values = tf.expand_dims(interpolation_values, axis=1)

        # Compute all rotations
        all_rotations_fn = lambda rot: tf.roll(interpolation_values, shift=rot, axis=3)
        interpolation_values = tf.vectorized_map(all_rotations_fn, tf.range(self.all_rotations))

        # Compute convolution
        # Shape kernel: (                            n_kernel, n_radial, n_angular, new_dim, feature_dim)
        # Shape values: (n_rotations, n_gpc_systems,        1, n_radial, n_angular,          feature_dim)
        # Shape result: (n_rotations, n_gpc_systems, n_kernel, n_radial, n_angular,          new_dim    )
        result = tf.linalg.matvec(self.kernels, interpolation_values)
        result = result + self.bias

        # Sum over all kernels (2), radial (3) and angular (4) coordinates
        # Shape result: (n_rotations, n_gpc_systems, new_dim)
        return tf.reduce_sum(result, axis=[2, 3, 4])

    @tf.function
    def _interpolate(self, signal, bary_coords):
        """

        """

        # Reshape for broadcasting convenience
        bary_coords = tf.reshape(bary_coords, (-1, 3, 2))

        # Gather vertex signals and weight them with Barycentric coordinates
        vertex_signals = tf.gather(signal, tf.cast(bary_coords[:, :, 0], tf.int32))
        vertex_signals = tf.multiply(vertex_signals, tf.expand_dims(bary_coords[:, :, 1], axis=-1))

        # Compute interpolation and reshape back to original shape
        interpolations = tf.math.reduce_sum(vertex_signals, axis=1)
        interpolations = tf.reshape(interpolations, (self.kernel_size[0], self.kernel_size[1], -1))
        return interpolations
