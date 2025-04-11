import tensorflow as tf


class NormalizePointCloud(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        # Move point-cloud into origin
        inputs = inputs - tf.reduce_mean(inputs, axis=1, keepdims=True)

        # Get axis-aligned bounding box
        aabb = tf.reduce_max(inputs, axis=1) - tf.reduce_min(inputs, axis=1)

        # Scale point-cloud so that largest axis has length 1
        return inputs / tf.reduce_max(aabb), aabb
