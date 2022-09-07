from tensorflow.keras.layers import Layer

import tensorflow as tf


class AngularMaxPooling(Layer):
    """Angular maximum pooling to filter for the maximal response of a geodesic convolution.

    The signals are compared vertex-wise at the hand of their norms.

    **Input**

    - The result of a geodesic convolution for each rotation in a tensor of size `(None, r, o)` where
      `r` the amount of rotations and `o` the output dimension of the convolution.

    **Output**

    - A tensor containing the maximal response at each vertex. Compare Eq. (12) in [1].

    [1]:
    > Jonathan Masci, Davide Boscaini, Michael M. Bronstein, Pierre Vandergheynst

    > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://www.cv-foundation.org/
    openaccess/content_iccv_2015_workshops/w22/html/Masci_Geodesic_Convolutional_Neural_ICCV_2015_paper.html)

    """

    @tf.function
    def call(self, inputs):
        rotation_norms = tf.norm(inputs, ord="euclidean", axis=-1)
        winner_rotation = tf.cast(tf.argmax(rotation_norms, axis=1), dtype=tf.int32)

        shape = tf.shape(winner_rotation)
        batch_size = shape[0]
        amt_gpc_systems = shape[1]
        batch_indices = tf.reshape(tf.repeat(tf.range(batch_size), repeats=amt_gpc_systems), (batch_size, -1))
        gpc_system_indices = tf.repeat(tf.expand_dims(tf.range(amt_gpc_systems), axis=0), repeats=batch_size, axis=0)
        indices = tf.stack([batch_indices, winner_rotation, gpc_system_indices], axis=-1)

        return tf.gather_nd(params=inputs, indices=indices)
