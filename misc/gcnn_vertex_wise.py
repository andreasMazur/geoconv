from tensorflow.keras.layers import Layer, Activation

import tensorflow as tf


@tf.function
def angular_max_pooling(signal):
    """Angular maximum pooling to filter for the maximal response of a geodesic convolution.

    The signals are measured at the hand of their norms.

    **Input**

    - The result of a geodesic convolution for each rotation in a tensor of size `(r, o)`, where `r` the amount of
      rotations and `o` the output dimension of the convolution.

    **Output**

    - A tensor containing the maximal response. Compare Eq. (12) in [1].

    [1]:
    > Jonathan Masci, Davide Boscaini, Michael M. Bronstein, Pierre Vandergheynst

    > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://www.cv-foundation.org/
    openaccess/content_iccv_2015_workshops/w22/html/Masci_Geodesic_Convolutional_Neural_ICCV_2015_paper.html)

    """
    max_response_idx = tf.argmax(tf.norm(signal, ord="euclidean", axis=-1), axis=-1)
    return signal[max_response_idx]


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
            shape=(self.kernel_size[0], self.kernel_size[1], self.amt_kernel, self.output_dim, signal_shape[-1]),
            initializer="glorot_uniform",
            trainable=True
        )

    @tf.function
    def call(self, inputs):
        """The geodesic convolution Layer performs a geodesic convolution.

        This layer computes the geodesic convolution for one vertex `v`:

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
            * A tensor containing the signal from the patch around the target vertex `v`. It has size `(r, a, 3, i)`
             where `r` and `a` reference the amount radial- and angular coordinates in the used kernel and `i` the input
             feature dimension. In particular, `inputs[0, x, y]` contains the 3 feature vectors which are used in
             combination with `inputs[1, x, y]` to compute the pullback-vector for kernel vertex `(x, y)`.
            * A tensor containing the Barycentric coordinates corresponding to the kernel defined for this layer.
              It has size `(r, a, 3)`. In particular, `inputs[1, x, y]` contains the 3 barycentric coordinates
              which are used in combination with `inputs[0, x, y]` to compute the pullback-vector for kernel vertex
              `(x, y)`.

        **Output**

        - A tensor of size `(o,)` representing the geodesic convolution of the signal with the layer's kernel. Here,
          `o` is the desired output dimension of the signal.

        """

        signal_batches, bary_c_batches = inputs
        result_tensor = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        batch_size = tf.shape(signal_batches)[0]
        for idx in tf.range(batch_size):
            # signal shape: (amt_radial_c, amt_angular_c, 3, input_dim)
            signal, bary_c = signal_batches[idx], bary_c_batches[idx]

            # signal shape: (amt_radial_c * amt_angular_c, 3, input_dim)
            signal = tf.reshape(signal, (-1, 3, self.kernel.shape[-1]))
            bary_c = tf.reshape(bary_c, (-1, 3))

            # signal shape: (amt_radial_c * amt_angular_c, input_dim, 3)
            signal = tf.vectorized_map(tf.transpose, signal)

            # pullback shape: (amt_radial_c * amt_angular_c, input_dim)
            pullback = tf.linalg.matvec(signal, bary_c)
            gc = lambda rot: self._geodesic_conv(pullback, rot)
            new_signal = tf.map_fn(
                gc,
                self.all_rotations,
                fn_output_signature=tf.TensorSpec([self.output_dim], dtype=tf.float32)
            )
            new_signal = self.activation(new_signal)
            # Angular max pooling over all rotations
            new_signal = angular_max_pooling(new_signal)
            result_tensor = result_tensor.write(idx, new_signal)
        return result_tensor.stack()

    @tf.function
    def _geodesic_conv(self, pullback, rotation):
        """Computes the geodesic convolution for exactly one rotation of the kernel."""

        sum_ = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for idx in tf.range(self.kernel_size[0] * self.kernel_size[1]):
            angular_idx = tf.math.floormod(idx, self.kernel_size[1])
            radial_idx = tf.cast((idx - angular_idx) / self.kernel_size[1], dtype=tf.int32)

            # add kernel rotation
            angular_idx = tf.math.floormod(angular_idx + rotation, self.kernel_size[1])
            new_signal = tf.linalg.matvec(self.kernel[radial_idx, angular_idx], pullback[idx])
            sum_ = sum_.write(idx, new_signal)
        sum_ = sum_.stack()

        # Summation over all kernel vertices
        sum_ = tf.reduce_sum(sum_, axis=0)

        # Summation over all kernels
        return tf.reduce_sum(sum_, axis=0)
