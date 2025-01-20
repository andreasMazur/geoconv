import tensorflow as tf


class NormalizePointCloud(tf.keras.layers.Layer):
    def call(self, inputs):
        # Move point-cloud into origin
        inputs = inputs - tf.reduce_mean(inputs, axis=1, keepdims=True)

        # Scale maximal axis-variance of point-cloud to one. Scale other axes accordingly.
        return inputs / tf.reduce_max(tf.math.reduce_std(inputs, axis=1))
