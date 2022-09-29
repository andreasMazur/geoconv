from tensorflow.keras.layers import Layer, Activation

import tensorflow as tf


class ConvGeodesic(Layer):
    """The Tensorflow implementation of the geodesic convolution

    Paper, that introduced the geodesic convolution:
    > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://arxiv.org/abs/1501.06297)
    > Jonathan Masci and Davide Boscaini et al.

    Attributes
    ----------
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

    def __init__(self,
                 output_dim,
                 amt_kernel,
                 activation="relu",
                 name=None,
                 rotation_delta=1,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 initializer="glorot_uniform"):
        if name:
            super().__init__(name=name)
        else:
            super().__init__()

        self.activation = Activation(activation)
        self.output_dim = output_dim
        self.rotation_delta = rotation_delta
        self.amt_kernel = amt_kernel
        self.kernel_regularizer = kernel_regularizer
        # self.kernel_regularizer_outer = kernel_regularizer_outer
        self.bias_regularizer = bias_regularizer
        self.initializer = initializer

        # Attributes that depend on the data and are set automatically in build
        self._kernel_size = None  # (#radial, #angular)
        self._kernel = None
        self._outer_kernel = None
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
                "amt_kernel": self.amt_kernel,
                "kernel_regularizer": self.kernel_regularizer,
                # "kernel_regularizer_outer": self.kernel_regularizer_outer,
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
        self._kernel_size = (barycentric_shape[2], barycentric_shape[3])
        self._all_rotations = self._kernel_size[1]
        # Weights the contributions of the interpolations
        self._kernel = self.add_weight(
            name="geoconv_kernel",
            shape=(self._kernel_size[0], self._kernel_size[1], self.amt_kernel, self.output_dim, signal_shape[2]),
            initializer=self.initializer,
            trainable=True,
            regularizer=self.kernel_regularizer
        )
        # Maps output to wished dimension
        # self._outer_kernel = self.add_weight(
        #     name="geoconv_outer",
        #     shape=(self.amt_kernel, self.output_dim, signal_shape[2]),
        #     initializer=self.initializer,
        #     trainable=True,
        #     regularizer=self.kernel_regularizer_outer
        # )
        self._bias = self.add_weight(
            name="geoconv_bias",
            shape=(self.amt_kernel, self.output_dim),
            initializer=self.initializer,
            trainable=True,
            regularizer=self.bias_regularizer
        )

    # @tf.function
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
            every rotation. It has size (n_batch, n_rotations, n_gpc_systems, feature_dim)
        """

        signal, b_coordinates = inputs
        result_tensor = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        batch_size = tf.shape(signal)[0]
        for idx in tf.range(batch_size):
            new_signal = self._geodesic_convolution(signal[idx], b_coordinates[idx])
            result_tensor = result_tensor.write(idx, new_signal)
        return result_tensor.stack()

    @tf.function
    def _geodesic_convolution(self, mesh_signal, barycentric_coords):

        # Interpolate signals at kernel vertices
        interpolation_fn = lambda bc: self._interpolate(mesh_signal, bc)
        mesh_signal = tf.vectorized_map(interpolation_fn, barycentric_coords)

        # Compute rotations
        all_rotations_fn = lambda rot: tf.roll(mesh_signal, shift=rot, axis=2)
        # n_rotations = ceil(self.all_rotations / self.rotation_delta)
        mesh_signal = tf.vectorized_map(
            all_rotations_fn, tf.range(start=0, limit=self._all_rotations, delta=self.rotation_delta)
        )
        mesh_signal = tf.expand_dims(mesh_signal, axis=4)

        # Compute convolution
        # Shape kernel: (                            n_radial, n_angular, n_kernel, self.output_dim, input_dim)
        # Shape input : (n_rotations, n_gpc_systems, n_radial, n_angular,        1,                  input_dim)
        # Shape result: (n_rotations, n_gpc_systems, n_radial, n_angular, n_kernel, self.output_dim           )
        mesh_signal = tf.reduce_sum(tf.linalg.matvec(self._kernel, mesh_signal) + self._bias, axis=[2, 3, 4])

        # Sum over all kernels (2)
        # Shape result: (n_rotations, n_gpc_systems, self.output_dim)
        return mesh_signal

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
