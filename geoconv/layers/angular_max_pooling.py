from tensorflow.keras.layers import Layer

import tensorflow as tf


class AngularMaxPooling(Layer):
    """

    """

    @tf.function
    def call(self, inputs):
        maximal_response = tf.norm(inputs, ord="euclidean", axis=-1)
        maximal_response = tf.cast(tf.argmax(maximal_response, axis=1), dtype=tf.int32)

        return tf.gather_nd(
            indices=tf.stack([tf.range(tf.shape(inputs)[0]), maximal_response], axis=-1),
            params=inputs
        )
