import tensorflow as tf


class LiftFeatures2D(tf.keras.layers.Layer):
    """Lifts features into 2-dimensional space by concatenating zeros after each given scalar."""
    @tf.function
    def call(self, inputs):
        """Applies the layer to the inputs.

        Parameters
        ----------
        inputs: tf.Tensor
            A tensor having a shape 'b x n x i', whereby 'b' refers to the number of shapes, 'n' to the number of
            vertices per shape and 'i' to the feature dimensionality.

        Returns
        -------
        tf.Tensor
            A tensor of shape 'b x n x 2i'.
        """
        input_shape = tf.shape(inputs)
        return tf.reshape(
            tf.stack([inputs, tf.zeros_like(inputs)], axis=-1),
            (input_shape[0], input_shape[1], input_shape[2] * 2)
        )
