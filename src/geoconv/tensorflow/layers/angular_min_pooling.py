import tensorflow as tf
import keras


class AngularMinPooling(keras.layers.Layer):
    """The implementation for angular max-pooling"""

    @tf.function
    def call(self, inputs):
        """Min-pools over the results of a intrinsic surface convolution.

        Parameters
        ----------
        inputs: tensorflow.Tensor
            The result tensor of an intrinsic surface convolution.
            It has a size of: (n_vertices, n_rotations, feature_dim), where 'n_vertices' references to the total amount
            of vertices in the triangle mesh, 'n_rotations' to the amount of rotations considered during the intrinsic
            surface convolution and 'feature_dim' to the feature dimensionality.

        Returns
        -------
        tensorflow.Tensor:
            A two-dimensional tensor of size (n_vertices, feature_dim), that contains a convolution results for each
            vertex. Thereby, the convolution result has the smallest Euclidean norm among the convolution results for
            all rotations.
        """
        minimal_response = tf.norm(inputs, ord="euclidean", axis=-1)
        minimal_response = tf.cast(tf.argmin(minimal_response, axis=1), dtype=tf.int32)
        return tf.gather_nd(
            indices=tf.stack([tf.range(tf.shape(inputs)[0]), minimal_response], axis=-1),
            params=inputs
        )
