import tensorflow as tf


def gravity(coordinates, t=1., delta=1.):
    # Compute centroid
    centroid = tf.reduce_mean(coordinates, axis=0)

    # Compute direction and distance to centroid
    directions = centroid - coordinates
    distances = tf.linalg.norm(directions, axis=-1)
    directions = directions / distances[:, None]

    # Zero gravity for points on delta-sphere
    new_coordinates = coordinates + (tf.math.square(t) * (distances - delta) / 2)[:, None] * directions

    return new_coordinates


class GravityPooling(tf.keras.layers.Layer):
    def __init__(self, delta=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta

    def call(self, inputs, *args, **kwargs):
        coordinates, t, iterations = inputs
        for _ in tf.range(iterations):
            coordinates = tf.map_fn(lambda x: gravity(x, t=t, delta=self.delta), coordinates, dtype=tf.float32)
        return coordinates
