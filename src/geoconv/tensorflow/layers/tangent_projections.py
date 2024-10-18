from geoconv.tensorflow.utils.compute_shot_lrf import (
    compute_distance_matrix,
    group_neighborhoods,
    shot_lrf,
    logarithmic_map
)

import tensorflow as tf


class TangentProjections(tf.keras.layers.Layer):
    def __init__(self, n_neighbors, neighbor_limit=20, n_bins=5):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.neighbor_limit = neighbor_limit
        self.n_bins = n_bins

    @tf.function
    def call(self, batched_coordinates):
        return tf.map_fn(self.call_helper, batched_coordinates)

    @tf.function
    def call_helper(self, coordinates):
        distance_matrix = compute_distance_matrix(coordinates)
        radius = tf.reduce_mean(
            tf.gather(distance_matrix, tf.argsort(distance_matrix, axis=-1)[:, self.n_neighbors], batch_dims=1)
        )

        # 1.) Get vertex-neighborhoods
        # 'neighborhoods': (vertices, n_neighbors, 3)
        neighborhoods, neighborhoods_indices = group_neighborhoods(
            coordinates,
            radius,
            neighbor_limit=self.neighbor_limit,
            distance_matrix=distance_matrix,
            fill_coordinate_length=radius*2  # Remove projections which are too far away from origin
        )

        # 2.) Get local reference frames
        # 'lrfs': (vertices, 3, 3)
        lrfs = shot_lrf(neighborhoods, radius)

        # 3.) Project neighborhoods into their lrfs using the logarithmic map
        # 'projections': (vertices, n_neighbors, 2)
        projections = logarithmic_map(lrfs, neighborhoods)

        # 4.) Create grid in projection spaces
        grid_axes = tf.linspace(start=-radius, stop=radius, num=self.n_bins + 1)
        step_size = 2 * radius / self.n_bins

        # 5.) Compute grid-weights
        center_points = grid_axes[:-1] + (step_size / 2)
        center_points = tf.stack(tf.meshgrid(center_points, center_points), axis=-1)
        weights = radius - tf.linalg.norm(center_points, axis=-1)

        def to_histogram(local_proj):
            # Put projections into bins
            bins = tf.math.floor((local_proj + radius) / step_size)

            # Filter out points that are within our grid
            bins = tf.gather(bins, tf.squeeze(tf.where(tf.math.reduce_all(bins < self.n_bins, axis=-1)), axis=-1))

            # Hash for counting
            bins = bins[:, 1] * 10 + bins[:, 0]

            # Count
            bins, _, counts = tf.unique_with_counts(bins)

            # Un-hash to get indices
            bins = tf.stack(
                [tf.cast(tf.math.floor(bins / 10), tf.int32), tf.cast(tf.math.floormod(bins, 10), tf.int32)],
                axis=-1
            )

            # Return histogram
            histogram = tf.tensor_scatter_nd_update(
                tf.zeros((self.n_bins, self.n_bins)), bins, tf.cast(counts, tf.float32)
            ) * weights
            return histogram / tf.linalg.norm(histogram)

        histograms = tf.map_fn(to_histogram, projections)
        result = tf.reshape(histograms, (tf.shape(coordinates)[0], self.n_bins ** 2))

        return result
