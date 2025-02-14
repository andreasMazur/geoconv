from geoconv.tensorflow.utils.compute_shot_lrf import compute_distance_matrix, compute_neighborhood

import tensorflow as tf


@tf.function(jit_compile=True)
def gravity(coordinates, t=1., delta=1.):
    # Compute centroid
    centroid = tf.reduce_mean(coordinates, axis=1)

    # Compute direction and distance to centroid
    directions = centroid[:, None, :] - coordinates
    distances = tf.linalg.norm(directions, axis=-1)
    directions = directions / distances[..., None]

    # Zero gravity for points on delta-sphere
    new_coordinates = coordinates + (tf.math.square(t) / 2 * (distances - delta))[..., None] * directions

    return new_coordinates


class GravityPooling(tf.keras.layers.Layer):
    def __init__(self, delta=1., neighbors_for_density=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.neighbors_for_density = neighbors_for_density

    def call(self, inputs, *args, **kwargs):
        coordinates, signal, t, iterations, n_vertices = inputs

        # gravity shift
        for _ in tf.range(iterations):
            coordinates = gravity(coordinates, t=t, delta=self.delta)

        # pooling
        if n_vertices == tf.shape(coordinates)[1]:
            return coordinates, signal
        else:
            result = tf.map_fn(lambda cs: self.pool(cs, n_vertices), tf.concat([coordinates, signal], axis=-1))
            return result[..., :3], result[..., 3:]

    @tf.function(jit_compile=True)
    def pool(self, coordinates_and_signal, n_vertices):
        # 'coordinates': (old_n_vertices, 3)
        coordinates = coordinates_and_signal[:, :3]
        # 'signal': (old_n_vertices, d)
        signal = coordinates_and_signal[:, 3:]
        neighborhoods, neighborhood_indices, radii = compute_neighborhood(coordinates, self.neighbors_for_density)
        # 'distance_matrices': (old_n_vertices, n_neighbors, n_neighbors)
        distance_matrices = tf.map_fn(compute_distance_matrix, neighborhoods)
        # 'densities': (old_n_vertices,)
        densities = tf.reduce_sum(distance_matrices, axis=[1, 2])
        # 'maintain': (n_vertices,)
        maintain = tf.argsort(densities, axis=-1)[:n_vertices]
        # 'neighborhood_indices': (n_vertices, self.neighbors_for_density)
        neighborhood_indices = tf.gather(neighborhood_indices, maintain)
        # 'signal': (n_vertices, self.neighbors_for_density, d)
        signal = tf.gather(signal, neighborhood_indices)
        # (n_vertices, 3 + d)
        return tf.concat([tf.gather(coordinates, maintain), tf.reduce_mean(signal, axis=1)], axis=-1)
