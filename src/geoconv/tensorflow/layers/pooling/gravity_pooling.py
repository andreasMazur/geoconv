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
    def __init__(self, n_vertices, iterations, time_span, delta=1., neighbors_for_density=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.neighbors_for_density = neighbors_for_density
        self.n_vertices = n_vertices
        self.iterations = iterations
        self.time_span = time_span

    def call(self, inputs, *args, **kwargs):
        coordinates, signal = inputs

        ################
        # Gravity Shift
        ################
        for _ in tf.range(self.iterations):
            coordinates = gravity(coordinates, t=self.time_span, delta=self.delta)

        ##########
        # Pooling
        ##########
        if self.n_vertices == tf.shape(coordinates)[1]:
            return coordinates, signal
        else:
            neighborhoods, neighborhood_indices, radii = compute_neighborhood(coordinates, self.neighbors_for_density)
            orig_shape = tf.shape(neighborhoods)

            # 'distance_matrices': (batch_size * old_n_vertices, self.neighbors_for_density, self.neighbors_for_density)
            distance_matrices = tf.map_fn(
                compute_distance_matrix, tf.reshape(neighborhoods, (-1, self.neighbors_for_density, 3))
            )
            # 'distance_matrices':  (batch_size, old_n_vertices, self.neighbors_for_density, self.neighbors_for_density)
            distance_matrices = tf.reshape(
                distance_matrices,
                (orig_shape[0], orig_shape[1], self.neighbors_for_density, self.neighbors_for_density)
            )
            # 'densities': (batch_size, old_n_vertices)
            densities = tf.reduce_sum(distance_matrices, axis=[-1, -2])
            # 'keep': (batch_size, self.n_vertices)
            keep = tf.argsort(densities, axis=-1)[:, :self.n_vertices]
            # 'neighborhood_indices': (batch_size, self.n_vertices, self.neighbors_for_density)
            neighborhood_indices = tf.gather(neighborhood_indices, keep, batch_dims=1)
            # 'signal': (batch_size, self.n_vertices, feature_dim)
            signal = tf.reduce_mean(tf.gather(signal, neighborhood_indices, batch_dims=1), axis=-2)
            # 'coordinates':  (batch_size, self.n_vertices, 3)
            coordinates = tf.gather(coordinates, keep, batch_dims=1)
            return coordinates, signal
