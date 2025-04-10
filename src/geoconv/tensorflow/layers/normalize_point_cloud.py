import tensorflow as tf
import numpy as np


class NormalizePointCloud(tf.keras.layers.Layer):
    def __init__(self):
        super(NormalizePointCloud, self).__init__()
        self.normalization_value = None

    def call(self, inputs, *args, **kwargs):
        # Move point-cloud into origin
        inputs = inputs - tf.reduce_mean(inputs, axis=1, keepdims=True)

        # Scale point-cloud with largest axis contained adaption dataset
        return inputs / self.normalization_value

    def adapt(self, dataset):
        # Find the largest axis in any point-cloud
        max_x_len, max_y_len, max_z_len = -np.inf, -np.inf, -np.inf
        for vertices, _ in dataset:
            x_len, y_len, z_len = vertices[0].numpy().max(axis=0) - vertices[0].numpy().min(axis=0)
            max_x_len = max(max_x_len, x_len)
            max_y_len = max(max_y_len, y_len)
            max_z_len = max(max_z_len, z_len)

        # Remember largest axis
        if max_x_len >= max_y_len and max_x_len >= max_z_len:
            self.normalization_value = max_x_len
        elif max_y_len >= max_x_len and max_y_len >= max_z_len:
            self.normalization_value = max_y_len
        else:
            self.normalization_value = max_z_len
