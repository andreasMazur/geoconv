from tensorflow.keras.layers import Layer, Activation

import tensorflow as tf


def signal_pullback(signal, barycentric_coords):
    """Computes the pullback of signals onto the position of a kernel vertex."""

    fst_weighted_sig = barycentric_coords[2] * signal[barycentric_coords[3]]
    snd_weighted_sig = barycentric_coords[4] * signal[barycentric_coords[5]]
    thr_weighted_sig = barycentric_coords[6] * signal[barycentric_coords[7]]

    return tf.reduce_sum([fst_weighted_sig, snd_weighted_sig, thr_weighted_sig], axis=0)


def geodesic_conv(signal, barycentric_coords_gpc, kernel, rotation):
    """Computes the geodesic convolution within a given local GPC."""

    products = []
    for barycentric_coords in barycentric_coords_gpc:
        pullback = signal_pullback(signal, barycentric_coords)
        products.append(kernel[barycentric_coords[0], barycentric_coords[1] + rotation] * pullback)

    return tf.reduce_sum(products, axis=0)


class ConvGeodesic(Layer):

    def __init__(self, kernel_size, output_dim, barycentric_coordinates, activation="relu"):
        super().__init__()
        self.kernel_size = kernel_size  # (#radial, #angular)
        self.output_dim = output_dim  # dimension of the output signal
        self.kernel = None
        self.barycentric_coordinates = barycentric_coordinates
        self.activation = Activation(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "Kernel",
            shape=(self.kernel_size[0], self.kernel_size[1], input_shape[0], self.output_dim),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        """The geodesic convolution Layer performs a geodesic convolution.

        The geodesic convolution as described in [1] and [2]. In particular, this layer computes:

            (f (*) K)[v] = activation(max_k { sum_ijw: K[i, (j+k) % N_theta] * x[i, j] })

            x[i, j] = E[v, i, j, x1] * f[x1] + E[v, i, j, x2] * f[x2] + E[v, i, j, x3] * f[x3]

        With K[i, j] containing the weight-matrix for a kernel vertex and x[i, j] being the interpolated signal at the
        kernel vertex (i, j). Compare Equation (7) and (11) in [1] as well as section 4.4 in [2].

        [1]:
        > Jonathan Masci, Davide Boscaini, Michael M. Bronstein, Pierre Vandergheynst

        > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://www.cv-foundation.org/
        openaccess/content_iccv_2015_workshops/w22/html/Masci_Geodesic_Convolutional_Neural_ICCV_2015_paper.html)

        [2]:
        > Adrien Poulenard and Maks Ovsjanikov.

        > [Multi-directional geodesic neural networks via equivariant convolution.](https://dl.acm.org/doi/abs/10.1145/
        3272127.3275102)

        **Input**

        - The signal on the mesh.

        **Output**

        - The geodesic convolution of the signal with the layer's kernel.

        """

        all_rotations = tf.range(self.kernel.shape[1])
        new_signal = tf.zeros((inputs.shape[0], self.kernel.shape[-1]))
        for idx, barycentric_coords_gpc in enumerate(self.barycentric_coordinates):
            conv_fn = lambda rotation: geodesic_conv(inputs, barycentric_coords_gpc, self.kernel, rotation)
            convolutions = tf.vectorized_map(conv_fn, all_rotations)
            norms = tf.norm(convolutions, ord="euclidean", axis=-1)
            new_signal[idx] = convolutions[tf.argmax(norms)]

        return self.activation(new_signal)
