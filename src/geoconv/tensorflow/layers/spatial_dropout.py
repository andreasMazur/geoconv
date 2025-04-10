import tensorflow as tf


class SpatialDropout(tf.keras.layers.Layer):

    def __init__(self, rate):
        super().__init__()
        self.dropout_rate = rate

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False, **kwargs):
        """Mask out entire feature maps"""
        if training:
            input_shape = tf.shape(inputs)
            mask = tf.expand_dims(
                tf.random.categorical(
                    tf.tile(
                        tf.math.log([[self.dropout_rate, 1 - self.dropout_rate]]),
                        (input_shape[0], 1),
                    ),
                    input_shape[-1],
                ),
                axis=1,
            )
            return tf.cast(mask, tf.float32) * inputs
        else:
            return inputs
