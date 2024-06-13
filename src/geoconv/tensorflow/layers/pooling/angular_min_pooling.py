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
            It has a size of: (batch_shapes, n_vertices, n_rotations, feature_dim), where 'n_vertices' references to the
            total amount of vertices in the triangle mesh, 'n_rotations' to the amount of rotations considered during
            the intrinsic surface convolution and 'feature_dim' to the feature dimensionality.

        Returns
        -------
        tensorflow.Tensor:
            A two-dimensional tensor of size (batch_shapes, n_vertices, feature_dim).
        """
        return tf.reduce_min(inputs, axis=-2)
