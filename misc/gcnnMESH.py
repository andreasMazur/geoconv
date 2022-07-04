from tensorflow.keras.layers import Layer, Activation

import tensorflow as tf


@tf.function
def angular_max_pooling(signal):
    """Angular maximum pooling to filter for the maximal response of a geodesic convolution.

    The signals are measured at the hand of their norms.

    **Input**

    - The result of a geodesic convolution for each rotation in a tensor of size `(m, r, o)`, where `m` is the amount
      of nodes on the mesh, `r` the amount of rotations and `o` the output dimesion of the convolution.

    **Output**

    - A tensor containing the maximal response at each vertex. Compare Eq. (12) in [1].

    [1]:
    > Jonathan Masci, Davide Boscaini, Michael M. Bronstein, Pierre Vandergheynst

    > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://www.cv-foundation.org/
    openaccess/content_iccv_2015_workshops/w22/html/Masci_Geodesic_Convolutional_Neural_ICCV_2015_paper.html)

    """
    column_indices = tf.argmax(tf.norm(signal, ord="euclidean", axis=-1), axis=-1)
    row_indices = tf.range(tf.shape(signal)[0], dtype=tf.int64)
    indices = tf.stack([row_indices, column_indices], axis=1)
    return tf.gather_nd(signal, indices)


@tf.function
def signal_pullback(signal, barycentric_coords_vertex):
    """Computes the pullback of signals onto the position of a kernel vertex.

    **Input**

    - `signal`: A Tensor containing the signal on the graph. It has the size `(m, n)` where `m` is the amount of
      nodes in the graph and `n` the dimensionality of the signal on the graph.

    - `barycentric_coords_vertex`: The barycentric coordinates for one kernel vertex of one local GPC-system. The form
      should be of the one described in `barycentric_coords_local_gpc`.

    **Output**

    - A tensor representing the pullback of the signal onto the position of the kernel vertex. The dimensionality of the
      tensor corresponds to the dimensionality of the input signal.

    """

    fst_weighted_sig = barycentric_coords_vertex[2] * signal[tf.cast(barycentric_coords_vertex[3], tf.int32)]
    snd_weighted_sig = barycentric_coords_vertex[4] * signal[tf.cast(barycentric_coords_vertex[5], tf.int32)]
    thr_weighted_sig = barycentric_coords_vertex[6] * signal[tf.cast(barycentric_coords_vertex[7], tf.int32)]

    return tf.reduce_sum([fst_weighted_sig, snd_weighted_sig, thr_weighted_sig], axis=0)


class ConvGeodesic(Layer):

    def __init__(self, kernel_size, output_dim, amt_kernel, activation="relu"):
        super().__init__()

        # Define kernel attributes
        self.kernel = None
        self.kernel_size = kernel_size  # (#radial, #angular)

        # Define output attributes
        self.output_dim = output_dim  # dimension of the output signal
        self.activation = Activation(activation)

        # Define convolution attributes
        self.all_rotations = tf.range(self.kernel_size[1])
        self.amt_kernel = amt_kernel

    def get_config(self):
        config = super(ConvGeodesic, self).get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "output_dim": self.output_dim,
                "activation": self.activation,
                "all_rotations": self.all_rotations
            }
        )
        return config

    def build(self, input_shape):
        """Defines the kernel for the geodesic convolution layer.

        In one layer we have:
            * `self.amt_kernel`-many kernels (referred to as 'filters' in [1]).
            * With `(self.kernel_size[0], self.kernel_size[1])` we reference to a weight matrix corresponding to a
              kernel vertex.
            * With `(self.output_dim, input_shape[1])` we define the size of the weight matrix of a kernel vertex.
              With `self.output_dim` we modify the output dimensionality of the signal after the convolution.

        An expressive illustration of how these kernels are used during the convolution is given in Figure 3 in [1].

        [1]:
        > Jonathan Masci, Davide Boscaini, Michael M. Bronstein, Pierre Vandergheynst

        > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://www.cv-foundation.org/
        openaccess/content_iccv_2015_workshops/w22/html/Masci_Geodesic_Convolutional_Neural_ICCV_2015_paper.html)


        **Input**

        - `input_shape`: The shape of the tensor containing the signal on the graph.

        """
        signal_shape, _ = input_shape
        self.kernel = self.add_weight(
            "Kernel",
            shape=(self.kernel_size[0], self.kernel_size[1], self.amt_kernel, self.output_dim, signal_shape[2]),
            initializer="glorot_uniform",
            trainable=True
        )

    # @tf.function
    def call(self, inputs):
        """The geodesic convolution Layer performs a geodesic convolution.

        This layer computes the geodesic convolution for all vertices `v`:

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
              It has the size `(m, self.kernel_size[0] * self.kernel_size[1], 8)` and is structured like described in
              `barycentric_coordinates.barycentric_coords.barycentric_coords_local_gpc`.

        **Output**

        - A tensor of size `(m, o)` representing the geodesic convolution of the signal with the layer's kernel. Here,
          `m` is the amount of nodes in the graph and `o` is the desired output dimension of the signal.

        """

        # TODO: Point-wise application! Not entire mesh as input!
        signal, b_coordinates = inputs
        result_tensor = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        batch_size = tf.shape(signal)[0]
        for idx in tf.range(batch_size):
            call_fn = lambda barycentric_coords_gpc: self._rotations(signal[idx], barycentric_coords_gpc)
            new_signal = tf.map_fn(
                call_fn,
                b_coordinates[idx],
                fn_output_signature=tf.TensorSpec([self.all_rotations.shape[0], self.output_dim], dtype=tf.float32)
            )
            new_signal = self.activation(new_signal)
            # Angular max pooling over all rotations
            new_signal = angular_max_pooling(new_signal)

            result_tensor = result_tensor.write(idx, new_signal)
        return result_tensor.stack()

    @tf.function
    def _rotations(self, signal, barycentric_coords_gpc):
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

        conv_fn = lambda rotation: self._kernel_vertices(signal, barycentric_coords_gpc, rotation)
        convolutions = tf.map_fn(
            conv_fn,
            self.all_rotations,
            fn_output_signature=tf.TensorSpec([self.output_dim], dtype=tf.float32)
        )

        return convolutions

    @tf.function
    def _kernel_vertices(self, signal, barycentric_coords_gpc, rotation):
        """Computes the geodesic convolution for given rotation and barycentric coordinates of a local GPC-system.

        In essence this function computes for exactly one r:
            sum_ij: K[i, (j+r) % N_theta] * x[i, j]


        **Input**

        - `inputs`: A Tensor containing the signal on the graph. It has the size `(m, n)` where `m` is the amount of
          nodes in the graph and `n` the dimensionality of the signal on the graph.

        - `barycentric_coords_gpc`: The barycentric coordinates for each kernel vertex of one local GPC-system. The form
          should be of the one described in `barycentric_coords_local_gpc`.

        - `rotation`: The considered rotation for the kernel.

        **Output**

        - A tensor of size `o` where `o` is the desired output dimensionality of the convolved signal.

        """

        geodesic_conv_fn = lambda bary_c: self._geodesic_conv(signal, bary_c, rotation)
        products = tf.map_fn(
            geodesic_conv_fn,
            barycentric_coords_gpc,
            fn_output_signature=tf.TensorSpec([self.amt_kernel, self.output_dim], dtype=tf.float32)
        )

        # At this point, we compute the sum over every vertex-convolution for each kernel
        sum_ij_per_kernel = tf.reduce_sum(products, axis=0)

        # Equation (11) in [1]: Return the sum over all kernels
        return tf.reduce_sum(sum_ij_per_kernel, axis=0)

    @tf.function
    def _geodesic_conv(self, signal, barycentric_coords, rotation):
        """Computes the most inner part of the geodesic convolution within a given local GPC.

        In essence this function computes:
            K[i, (j+r) % N_theta] * x[i, j]

        with all variables (i, j, r) are already given at this point in the computation. Keep in mind that we calculate
        with `self.amt_kernel`-many `K` at once.

        **Input**

        - `inputs`: A Tensor containing the signal on the graph. It has the size `(m, n)` where `m` is the amount of
          nodes in the graph and `n` the dimensionality of the signal on the graph.

        - `barycentric_coords_gpc`: The barycentric coordinates for each kernel vertex of one local GPC-system. The form
          should be of the one described in `barycentric_coords_local_gpc`.

        - `rotation`: The considered rotation for the kernel.

        **Output**

        - A tensor of size `o` where `o` is the desired output dimensionality of the convolved signal.

        """

        pullback = signal_pullback(signal, barycentric_coords)
        radial_idx = tf.cast(barycentric_coords[0], tf.int32)
        angular_idx = tf.cast(barycentric_coords[1], tf.int32) + rotation
        angular_idx = tf.math.floormod(angular_idx, self.kernel_size[1])

        return tf.linalg.matvec(self.kernel[radial_idx, angular_idx], pullback)
