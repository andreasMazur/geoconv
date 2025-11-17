import tensorflow as tf


class GlobalComplexPooling(tf.keras.layers.Layer):
    @tf.function
    def call(self, inputs):
        """Takes the maximum value over each channel."""
        # inputs_shape contains the following values: (batch, vertices, input_dim)
        inputs_shape = tf.shape(inputs)

        # Reshape inputs into complex value structure
        # inputs : (batch, vertices, input_dim / 2, 2)
        inputs = tf.reshape(inputs, tf.concat([inputs_shape[:-1], [-1], [2]], axis=-1))

        # Determine amplitudes
        # amplitudes : (batch, vertices, input_dim / 2)
        amplitudes = tf.linalg.norm(inputs, axis=-1)

        # Determine largest amplitudes among all vertices
        # index_largest_amplitude : (batch, input_dim / 2)
        index_largest_amplitude = tf.argmax(amplitudes, axis=-2)

        # Gather complex values with the largest amplitudes
        # inputs : (batch, input_dim / 2, 2)
        n_complex = tf.math.floordiv(inputs_shape[-1], 2)
        channel_indices = tf.tile(tf.range(n_complex, dtype=tf.int64)[None, :], multiples=[inputs_shape[0], 1])
        inputs = tf.gather_nd(
            inputs,
            tf.stack([index_largest_amplitude, channel_indices], axis=-1),
            batch_dims=1
        )

        # Reshape to original shape
        # inputs : (batch, input_dim)
        return tf.reshape(inputs, (inputs_shape[0], inputs_shape[-1]))


if __name__ == "__main__":
    input_ = tf.random.uniform([2, 768, 32])
    GlobalComplexPooling()(input_)