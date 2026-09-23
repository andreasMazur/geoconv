import tensorflow as tf


class UnLiftFeatures2D(tf.keras.layers.Layer):
    """Lifts features into 2-dimensional space by concatenating zeros after each given scalar."""
    @tf.function
    def call(self, inputs):
        """Applies the layer to the inputs.
            
        Parameters
        ----------
        inputs: tf.Tensor
            The signal tensor which has shape 'b x n x 2i', whereby 'b' refers to the number of shapes, 'n' to the number
            of vertices per shape and 'i' to the feature dimensionality.

        Returns
        -------
        tf.Tensor
            A tensor of shape 'b x n x i'.
        """
        input_shape = tf.shape(inputs)
        return tf.reshape(
            inputs, (input_shape[0], input_shape[1], tf.math.floordiv(input_shape[-1], 2), 2)
        )[..., 0]
