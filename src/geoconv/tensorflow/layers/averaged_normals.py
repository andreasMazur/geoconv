from geoconv.preprocessing.barycentric_coordinates import create_template_matrix
from geoconv.tensorflow.utils.compute_shot_lrf import compute_distance_matrix, knn_shot_lrf

import tensorflow as tf


class AveragedNormals(tf.keras.layers.Layer):
    def __init__(self, n_radial, n_angular, neighbors_for_lrf=15, neighbors_for_avg=32, gamma=1, beta=1 / 4, alpha=10):
        super().__init__()
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.template = None

        self.neighbors_for_lrf = neighbors_for_lrf
        self.neighbors_for_avg = neighbors_for_avg

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def adapt(self, data=None, n_neighbors=None, template_scale=None, template_radius=None):
        """Sets the template radius to a given or the average neighborhood radius scaled by used defined coefficient.

        Parameters
        ----------
        data: tf.Dataset
            The training data which is used to compute the template radius.
        n_neighbors: int
            The amount of closest neighbors to consider to compute the template radius.
        template_scale: float
            The scaling factor to multiply on the template.
        template_radius: float
            The template radius to use to initialize the template.

        Returns
        -------
        float:
            The final template radius.
        """
        assert data is not None or template_radius is not None, "Must provide either 'data' or 'template_radius'."
        if None not in [data, n_neighbors, template_scale]:
            assert data is not None, "If 'template_radius' is not given, you must provide 'data'."
            assert n_neighbors is not None, "If 'template_radius' is not given, you must provide 'n_neighbors'."
            assert template_scale is not None, "If 'template_radius' is not given, you must provide 'template_scale'."

        # If no template radius is given, compute the template radius
        if template_radius is None:
            avg_radius, vertices_count = 0, 0
            for idx, (vertices, _) in enumerate(data):
                distance_matrix = compute_distance_matrix(tf.cast(vertices[0], tf.float32))
                radii = tf.gather(
                    distance_matrix, tf.argsort(distance_matrix, axis=-1)[:, n_neighbors], batch_dims=1
                )
                avg_radius = avg_radius + tf.reduce_sum(radii)
                vertices_count = vertices_count + tf.cast(tf.shape(radii)[0], tf.float32)
            avg_radius = avg_radius / vertices_count
            template_radius = avg_radius * template_scale

        # Initialize template
        self.template = tf.reshape(
            tf.constant(
                create_template_matrix(
                    n_radial=self.n_radial, n_angular=self.n_angular, radius=template_radius, in_cart=False
                ),
                dtype=tf.float32
            ),
            (self.n_radial, self.n_angular, 1, 1, 2)
        )

        # Return used template radius
        return template_radius

    @tf.function(jit_compile=True)
    def call(self, vertices):
        return tf.map_fn(self.call_helper, vertices)

    @tf.function(jit_compile=True)
    def call_helper(self, vertices):
        lrfs, _, neighborhoods_indices = knn_shot_lrf(self.neighbors_for_lrf, vertices)
        avg_normal = tf.reduce_mean(tf.gather(lrfs[:, 0, :], neighborhoods_indices[:, :self.neighbors_for_avg]), axis=1)
        return avg_normal / tf.expand_dims(tf.linalg.norm(avg_normal, axis=-1), axis=-1)
