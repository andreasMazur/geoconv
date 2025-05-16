import tensorflow as tf
import keras


class AngularMaxPooling(keras.layers.Layer):
    """The implementation for angular max-pooling"""

    @tf.function(jit_compile=True)
    def call(self, inputs):
        """Max-pools over the results of a intrinsic surface convolution.

        Parameters
        ----------
        inputs: tensorflow.Tensor
            A tensor of size: (batch_shapes, n_vertices, n_rotations, feature_dim), where 'n_vertices' references to the
            total amount of vertices in the triangle mesh, 'n_rotations' to the amount of rotations considered during
            the intrinsic surface convolution and 'feature_dim' to the feature dimensionality.

        Returns
        -------
        tensorflow.Tensor:
            A three-dimensional tensor of size (batch_shapes, n_vertices, feature_dim).
        """
        return tf.reduce_max(inputs, axis=-2)
        # return tf.gather(inputs, tf.argmax(tf.linalg.norm(inputs, ord="euclidean", axis=-1), axis=-1), batch_dims=2)
