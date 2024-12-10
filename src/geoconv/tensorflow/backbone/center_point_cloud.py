import tensorflow as tf


class CenterPointCloud(tf.keras.layers.Layer):

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs, axis=1, keepdims=True)
