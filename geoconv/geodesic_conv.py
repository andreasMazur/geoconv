from tensorflow.keras.layers import Layer, Activation

import tensorflow as tf


@tf.function
def angular_max_pooling(signal):
    """Angular maximum pooling to filter for the maximal response of a geodesic convolution.

    The signals are compared vertex-wise at the hand of their norms.

    **Input**

    - The result of a geodesic convolution for each rotation in a tensor of size `(r, o)` where
      `r` the amount of rotations and `o` the output dimension of the convolution.

    **Output**

    - A tensor containing the maximal response at each vertex. Compare Eq. (12) in [1].

    [1]:
    > Jonathan Masci, Davide Boscaini, Michael M. Bronstein, Pierre Vandergheynst

    > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://www.cv-foundation.org/
    openaccess/content_iccv_2015_workshops/w22/html/Masci_Geodesic_Convolutional_Neural_ICCV_2015_paper.html)

    """
    rotation_norms = tf.norm(signal, ord="euclidean", axis=-1)
    winner_rotation = tf.cast(tf.argmax(rotation_norms, axis=0), dtype=tf.int32)
    winner_rotation = tf.stack([winner_rotation, tf.range(winner_rotation.shape[0])], axis=-1)
    return tf.gather_nd(params=signal, indices=winner_rotation)


class ConvGeodesic(Layer):

    def __init__(self, kernel_size, output_dim, amt_kernel, activation="relu", name=None):
        if name:
            super().__init__(name=name)
        else:
            super().__init__()

        # Define kernel attributes
        self.kernels = []
        self.kernel_size = kernel_size  # (#radial, #angular)
        self.bias = None

        # Define output attributes
        self.output_dim = output_dim  # dimension of the output signal
        self.activation = Activation(activation)

        # Define convolution attributes
        self.all_rotations = self.kernel_size[1]
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
        """Defines the kernel for the geodesic convolution layer.

        In one layer we have:
            * `self.amt_kernel`-many kernels (referred to as 'filters' in [1]).
            * Each kernel defines `self.kernel_size[1]` (angular-coordinates-many) weight-tensors. Each of those
              describes weight-matrices for all `self-kernel_size[0]` (radial) coordinates.

        An expressive illustration of how these kernels are used during the convolution is given in Figure 3 in [1].

        [1]:
        > Jonathan Masci, Davide Boscaini, Michael M. Bronstein, Pierre Vandergheynst

        > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://www.cv-foundation.org/
        openaccess/content_iccv_2015_workshops/w22/html/Masci_Geodesic_Convolutional_Neural_ICCV_2015_paper.html)


        **Input**

        - `input_shape`: The shape of the tensor containing the signal on the graph.

        """
        signal_shape, _ = input_shape
        self.kernels = [
            [
                self.add_weight(
                    f"Kernel_{k}/AngularWeights_{a}",
                    shape=(self.kernel_size[0], self.output_dim, signal_shape[2]),
                    initializer="glorot_uniform",
                    trainable=True
                ) for a in range(self.kernel_size[1])
            ] for k in range(self.amt_kernel)
        ]
        self.bias = [
            self.add_weight(
                f"Bias_{k}", shape=(self.output_dim,),
                initializer="glorot_uniform",
                trainable=True
            ) for k in range(self.amt_kernel)
        ]

    @tf.function
    def call(self, inputs):
        """The geodesic convolution Layer performs a geodesic convolution.

        This layer computes the geodesic convolution with angular max pooling for all vertices `v`:

            (f (*) K)[v] = max_r{ activation(sum_ij: K[i, (j+r) % N_theta] * x[i, j]) }

            x[i, j] = E[v, i, j, x1] * f[x1] + E[v, i, j, x2] * f[x2] + E[v, i, j, x3] * f[x3]

        With K[i, j] containing the weight-matrix for a kernel vertex and x[i, j] being the interpolated signal at the
        kernel vertex (i, j). Furthermore, x1, x2 and x3 are the nodes in the mesh used to interpolate the signal at
        the kernel vertex (i, j) [usually the triangle including it]. E contains the necessary Barycentric coordinates
        for the interpolation. Compare Equation (7) and (11) in [1] as well as section 4.4 in [2].

        [1]:
        > Jonathan Masci, Davide Boscaini, Michael M. Bronstein, Pierre Vandergheynst

        > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://www.cv-foundation.org/
        openaccess/content_iccv_2015_workshops/w22/html/Masci_Geodesic_Convolutional_Neural_ICCV_2015_paper.html)

        [2]:
        > Adrien Poulenard and Maks Ovsjanikov.

        > [Multi-directional geodesic neural networks via equivariant convolution.](https://dl.acm.org/doi/abs/10.1145/
        3272127.3275102)

        **Input**

        - `inputs`:
            * A tensor containing the signal on the graph. It has the size `(m, n)` where `m` is the amount of
              nodes in the graph and `n` the dimensionality of the signal on the graph.
            * A tensor containing the Barycentric coordinates corresponding to the kernel defined for this layer.

        **Output**

        - A tensor of size `(m, o)` representing the geodesic convolution of the signal with the layer's kernel. Here,
          `m` is the amount of nodes in the graph and `o` is the desired output dimension of the signal.

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
        """Wrapper for computing the geodesic convolution w.r.t. each rotation.

        In essence this function computes for all r in self.all_rotations:
            sum_ij: K[i, (j+r) % N_theta] * x[i, j]

        **Input**

        - `inputs`: A Tensor containing the signal on the graph. It has the size `(m, n)` where `m` is the amount of
          nodes in the graph and `n` the dimensionality of the signal on the graph.

        - `barycentric_coords_gpc`: The barycentric coordinates for each kernel vertex of one local GPC-system. The form
          should be of the one described in `barycentric_coords_local_gpc`.

        **Output**

        - A tensor of size `(p, o)` where `p` is the amount of all considered rotations and `o` is the desired output
          dimensionality of the convolved signal.

        """

        # Interpolate signals at kernel vertices
        interpolation_fn = lambda bc: self.interpolate(signal, bc)
        interpolation_values = tf.vectorized_map(interpolation_fn, barycentric_coords)

        # We will need to sum over the convolutions of every kernel
        kernel_results = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        kernel_idx = tf.constant(0)
        for kernel in self.kernels:
            angular_results = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            angular_coordinate = tf.constant(0)
            for angular_weights in kernel:
                rotation_results = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
                for rotation in tf.range(self.all_rotations):
                    rotation = tf.math.floormod(angular_coordinate + rotation, self.kernel_size[1])
                    # (amt_gpc_systems, amt_radial_coords, output_dim)
                    result = tf.linalg.matvec(angular_weights, interpolation_values[:, rotation])
                    rotation_results = rotation_results.write(rotation, result)
                # (amt_rotations, amt_gpc_systems, amt_radial_coords, output_dim)
                rotation_results = rotation_results.stack()
                angular_results = angular_results.write(angular_coordinate, rotation_results)
                angular_coordinate = angular_coordinate + tf.constant(1)
            # (amt_angular_coords, amt_rotations, amt_gpc_systems, amt_radial_coords, output_dim)
            angular_results = angular_results.stack()
            kernel_results = kernel_results.write(kernel_idx, angular_results)
            kernel_idx = kernel_idx + tf.constant(1)

        idx = tf.constant(0)
        for bias in self.bias:
            bias_added = kernel_results.read(idx) + bias
            kernel_results = kernel_results.write(idx, bias_added)
            idx = idx + tf.constant(1)

        # (amt_kernel, amt_angular_coords, amt_rotations, amt_gpc_systems, amt_radial_coords, output_dim)
        kernel_results = kernel_results.stack()

        # Compute the sum over all filter kernels (0), as well as angular (1) and radial (4) coordinates
        # => (amt_rotations, amt_gpc_systems, output_dim)
        kernel_results = tf.reduce_sum(kernel_results, axis=[0, 1, 4])
        kernel_results = self.activation(kernel_results)

        return angular_max_pooling(kernel_results)

    @tf.function
    def interpolate(self, signal, bary_coords):
        """Computes the interpolation of signals in the position of a kernel vertex

        **Input**

        - `signal`: A Tensor containing the signal on the graph. It has the size `(m, n)` where `m` is the amount of
          nodes in the graph and `n` the dimensionality of the signal on the graph.

        - `barycentric_coords_vertex`: The barycentric coordinates for all kernel vertices of one local GPC-system. The
           form should be of the one described in `barycentric_coordinates`.

        **Output**

        - A tensor representing the pullback of the signal onto the position of the kernel vertex. The dimensionality of
          the tensor corresponds to the dimensionality of the input signal.
        """

        # Reshape for broadcasting convenience
        bary_coords = tf.reshape(bary_coords, (-1, 3, 2))

        # Gather vertex signals and weight them with Barycentric coordinates
        vertex_signals = tf.gather(signal, tf.cast(bary_coords[:, :, 0], tf.int32))
        vertex_signals = tf.multiply(vertex_signals, tf.expand_dims(bary_coords[:, :, 1], axis=-1))

        # Compute interpolation and reshape back to original shape
        interpolations = tf.math.reduce_sum(vertex_signals, axis=1)
        interpolations = tf.reshape(interpolations, (self.kernel_size[1], self.kernel_size[0], -1))
        return interpolations
