import tensorflow as tf

class BetaRelu(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            name="beta",
            shape=(input_shape[-1] // 2,),
            trainable=True
        )

    @tf.function
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, (input_shape[0], input_shape[1], input_shape[2] // 2, 2))
        inputs_norm = tf.linalg.norm(inputs, axis=-1)
        inputs = tf.nn.relu(inputs_norm - self.beta)[..., None] * tf.math.divide_no_nan(inputs, inputs_norm[..., None])
        return tf.reshape(inputs, (input_shape[0], input_shape[1], input_shape[2]))

