import tensorflow as tf


class UnLiftFeatures2D(tf.keras.layers.Layer):
    """Lifts features into 2-dimensional space by concatenating zeros after each given scalar."""
    @tf.function
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        return tf.reshape(
            inputs, (input_shape[0], input_shape[1], tf.math.floordiv(input_shape[-1], 2), 2)
        )[..., 0]
