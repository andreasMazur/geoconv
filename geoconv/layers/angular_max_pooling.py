from tensorflow import keras

import tensorflow as tf


class AngularMaxPooling(keras.layers.Layer):
    """The implementation of the geodesic convolution

    Paper, that introduced angular max-pooling:
    > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://arxiv.org/abs/1501.06297)
    > Jonathan Masci and Davide Boscaini et al.
    """

    @tf.function
    def call(self, inputs):
        """Max-pools over the results of a geodesic convolution.

        Parameters
        ----------
        inputs: tf.Tensor
            The result tensor of a geodesic convolution. It has a size of: (n_vertices, n_rotations, feature_dim),
            where 'n_vertices' references to the total amount of vertices in the triangle mesh, 'n_rotations' to the
            amount of rotations conducted in the geodesic convolution and 'feature_dim' to the feature dimensionality.
        """
        maximal_response = tf.norm(inputs, ord="euclidean", axis=-1)
        maximal_response = tf.cast(tf.argmax(maximal_response, axis=1), dtype=tf.int32)

        return tf.gather_nd(
            indices=tf.stack([tf.range(tf.shape(inputs)[0]), maximal_response], axis=-1),
            params=inputs
        )
