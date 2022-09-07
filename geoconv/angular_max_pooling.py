from tensorflow.keras.layers import Layer

import tensorflow as tf


class AngularMaxPooling(Layer):
    """

    """

    @tf.function
    def call(self, inputs):
        rotation_norms = tf.norm(inputs, ord="euclidean", axis=-1)
        winner_rotation = tf.cast(tf.argmax(rotation_norms, axis=1), dtype=tf.int32)

        shape = tf.shape(winner_rotation)
        batch_size = shape[0]
        amt_gpc_systems = shape[1]
        batch_indices = tf.reshape(tf.repeat(tf.range(batch_size), repeats=amt_gpc_systems), (batch_size, -1))
        gpc_system_indices = tf.repeat(tf.expand_dims(tf.range(amt_gpc_systems), axis=0), repeats=batch_size, axis=0)
        indices = tf.stack([batch_indices, winner_rotation, gpc_system_indices], axis=-1)

        return tf.gather_nd(params=inputs, indices=indices)
