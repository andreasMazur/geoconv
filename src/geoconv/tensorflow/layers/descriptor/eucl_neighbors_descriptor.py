from geoconv.tensorflow.utils.compute_shot_lrf import compute_neighborhood

import tensorflow as tf


class EuclNeighborsDescriptor(tf.keras.layers.Layer):
    def __init__(self, n_radial, n_angular):
        super().__init__()
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.n_neighbors = int(1 + self.n_radial * self.n_angular)

    def call(self, inputs, **kwargs):
        input_shape = tf.shape(inputs)

        # 'neighborhoods' : (batch, vertices, n_neighbors, 3)
        neighborhoods, _, _ = compute_neighborhood(inputs, self.n_neighbors)
        return tf.reshape(neighborhoods, (input_shape[0], input_shape[1], self.n_neighbors))
