from tensorflow.keras.layers import Layer

import tensorflow as tf


class AngularMaxPooling(Layer):
    """

    """

    @tf.function
    def call(self, inputs):

        return tf.vectorized_map(self._amp, inputs)

    @tf.function
    def _amp(self, mesh_signal):
        maximal_response = tf.norm(mesh_signal, ord="euclidean", axis=-1)
        maximal_response = tf.cast(tf.argmax(maximal_response, axis=1), dtype=tf.int32)

        return tf.gather_nd(
            indices=tf.stack([tf.range(tf.shape(mesh_signal)[0]), maximal_response], axis=-1),
            params=mesh_signal
        )
