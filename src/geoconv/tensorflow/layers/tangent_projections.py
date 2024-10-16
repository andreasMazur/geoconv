from geoconv.tensorflow.utils.compute_shot_lrf import (
    compute_distance_matrix,
    group_neighborhoods,
    shot_lrf,
    logarithmic_map
)

import tensorflow as tf


class TangentProjections(tf.keras.layers.Layer):
    def __init__(self, n_neighbors):
        super().__init__()
        self.n_neighbors = n_neighbors

    @tf.function(jit_compile=True)
    def call(self, batched_coordinates):
        return tf.map_fn(self.call_helper, batched_coordinates)

    @tf.function(jit_compile=True)
    def call_helper(self, coordinates):
        distance_matrix = compute_distance_matrix(coordinates)
        radii = tf.gather(distance_matrix, tf.argsort(distance_matrix, axis=-1)[:, self.n_neighbors], batch_dims=1)

        # 1.) Get vertex-neighborhoods
        # 'neighborhoods': (vertices, n_neighbors, 3)
        neighborhoods, neighborhoods_indices = group_neighborhoods(
            coordinates, radii, self.n_neighbors, distance_matrix
        )

        # 2.) Get local reference frames
        # 'lrfs': (vertices, 3, 3)
        lrfs = shot_lrf(neighborhoods, radii)

        # 3.) Project neighborhoods into their lrfs using the logarithmic map
        # 'projections = logarithmic_map(lrfs, neighborhoods)': (vertices, n_neighbors, 2)
        return logarithmic_map(lrfs, neighborhoods)
