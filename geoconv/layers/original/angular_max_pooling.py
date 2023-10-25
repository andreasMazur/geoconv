from tensorflow import keras

import tensorflow as tf


class AngularMaxPooling(keras.layers.Layer):
    """The implementation for angular max-pooling"""

    @tf.function
    def call(self, inputs):
        """Max-pools over the results of a geodesic convolution.

        Parameters
        ----------
        inputs: tf.Tensor
            The result tensor of a geodesic convolution. It has a size of: (n_vertices, n_rotations, feature_dim),
            where 'n_vertices' references to the total amount of vertices in the triangle mesh, 'n_rotations' to the
            amount of rotations conducted in the geodesic convolution and 'feature_dim' to the feature dimensionality.

        Returns
        -------
        tf.Tensor:
            A two-dimensional tensor of size (n_vertices, feature_dim), that contains the convolution result of the
            rotation that has the largest Euclidean norm.
        """
        maximal_response = tf.norm(inputs, ord="euclidean", axis=-1)
        maximal_response = tf.cast(tf.argmax(maximal_response, axis=1), dtype=tf.int32)

        return tf.gather_nd(
            indices=tf.stack([tf.range(tf.shape(inputs)[0]), maximal_response], axis=-1),
            params=inputs
        )
