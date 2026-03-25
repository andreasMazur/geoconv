from geoconv.tensorflow.utils.compute_shot_lrf import compute_neighborhood

import tensorflow as tf


class EuclNeighborsDescriptor(tf.keras.layers.Layer):
    def __init__(self, n_neighbors, normalize=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors
        self.normalize = normalize

    def call(self, inputs, **kwargs):
        # 'neighborhoods' : (batch, vertices, n_neighbors, 3)
        neighborhoods, _, _ = compute_neighborhood(inputs, self.n_neighbors)

        # Scale max sphere radius to 1
        if self.normalize:
            neighborhoods = neighborhoods / tf.reduce_max(tf.linalg.norm(neighborhoods, axis=-1), axis=-1)[..., None, None]
        neighborhoods_shape = tf.shape(neighborhoods)
        return tf.reshape(
            neighborhoods, (neighborhoods_shape[0], neighborhoods_shape[1], self.n_neighbors * 3)
        )[..., 3:]  # Cut away the origin-zero vectors

