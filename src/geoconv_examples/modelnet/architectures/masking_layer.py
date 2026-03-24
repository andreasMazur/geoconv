import tensorflow as tf


class MaskingLayer(tf.keras.layers.Layer):
    @tf.function
    def call(self, inputs):
        signal, mask = inputs
        return tf.expand_dims(signal[mask], axis=0)
