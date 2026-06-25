import tensorflow as tf


class LiftFeatures2D(tf.keras.layers.Layer):
    """Lifts features into 2-dimensional space by concatenating zeros after each given scalar."""
    @tf.function
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        return tf.reshape(
            tf.stack([inputs, tf.zeros_like(inputs)], axis=-1),
            (input_shape[0], input_shape[1], input_shape[2] * 2)
        )
