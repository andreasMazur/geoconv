from geoconv.preprocessing.barycentric_coordinates import create_template_matrix
from geoconv.tensorflow.utils.compute_shot_lrf import group_neighborhoods, shot_lrf, logarithmic_map, \
    compute_distance_matrix

import tensorflow as tf


class BarycentricCoordinates(tf.keras.layers.Layer):
    """A parameter-free neural network layer that approximates barycentric coordinates (BC).

    Attributes
    ----------
    n_radial: int
        The amount of radial coordinates of the template for which BC shall be computed.
    n_angular: int
        The amount of angular coordinates of the template for which BC shall be computed.
    radius: float
        The radius of the neighborhoods which are used for tangent-plane estimations.
    template_scale: float
        A scaling factor that is multiple onto the neighborhood radius in order to compute the template radius.
    """
    def __init__(self, n_radial, n_angular, n_neighbors, template_scale=0.75):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.template_scale = template_scale
        self.template = None

    def adapt(self, data=None, template_radius=None):
        """Sets the template radius to a given or the average neighborhood radius scaled by used defined coefficient.

        Parameters
        ----------
        data: tf.Dataset
            The training data which is used to compute the template radius.
        template_radius: float
            The template radius to use to initialize the template.

        Returns
        -------
        float:
            The final template radius.
        """
        assert data is not None or template_radius is not None, "Must provide either 'data' or 'template_radius'."

        # If no template radius is given, compute the
        if template_radius is None:
            avg_radius = 0
            for idx, vertices in enumerate(data):
                distance_matrix = compute_distance_matrix(tf.cast(vertices, tf.float32))
                radii = tf.gather(
                    distance_matrix, tf.argsort(distance_matrix, axis=-1)[:, self.n_neighbors], batch_dims=1
                )
                avg_radius = avg_radius + tf.reduce_mean(radii)
            avg_radius = avg_radius / (idx + 1)
            template_radius = avg_radius * self.template_scale

        # Initialize template
        self.template = tf.constant(
            create_template_matrix(
                n_radial=self.n_radial, n_angular=self.n_angular, radius=template_radius, in_cart=True
            ), dtype=tf.float32
        )

        # Return used template radius
        return template_radius

    @tf.function
    def call(self, vertices):
        """Computes barycentric coordinates for multiple shapes.

        Parameters
        ----------
        vertices: tf.Tensor
            A 3D-tensor of shape (batch_shapes, n_vertices, 3) that contains the vertices of the shapes.

        Returns
        -------
        tf.Tensor:
            A 5D-tensor of shape (batch_shapes, vertices, n_radial, n_angular, 3, 2) that describes barycentric
            coordinates.
        """
        return tf.map_fn(self.call_helper, vertices)

    @tf.function
    def call_helper(self, vertices):
        """Computes barycentric coordinates for a single shape.

        Parameters
        ----------
        vertices: tf.Tensor
            A 2D-tensor of shape (n_vertices, 3) that contains the vertices of the shapes.

        Returns
        -------
        tf.Tensor:
            A 4D-tensor of shape (vertices, n_radial, n_angular, 3, 2) that describes barycentric coordinates.
        """
        distance_matrix = compute_distance_matrix(vertices)
        radii = tf.gather(distance_matrix, tf.argsort(distance_matrix, axis=-1)[:, self.n_neighbors], batch_dims=1)

        # 1.) Get vertex-neighborhoods
        # 'neighborhoods': (vertices, n_neighbors, 3)
        neighborhoods, neighborhoods_indices = group_neighborhoods(vertices, radii, distance_matrix)

        # 2.) Get local reference frames
        # 'lrfs': (vertices, 3, 3)
        lrfs = shot_lrf(neighborhoods, radii)

        # 3.) Project neighborhoods into their lrfs using the logarithmic map
        # 'projections': (vertices, n_neighbors, 2)
        projections = logarithmic_map(lrfs, neighborhoods)

        # 4.) Compute barycentric coordinates
        # 'closest_proj': (vertices, 3, n_radial, n_angular)
        closest_proj = tf.argsort(
            tf.linalg.norm(self.template - tf.expand_dims(tf.expand_dims(projections, axis=2), axis=2), axis=-1), axis=1
        )[:, :3, :, :]

        # 'projections': (vertices, 3, n_radial, n_angular, 2)
        projections = tf.gather(projections, closest_proj, batch_dims=1)

        v0 = projections[:, 2] - projections[:, 0]
        v1 = projections[:, 1] - projections[:, 0]
        v2 = self.template - projections[:, 0]

        dot00 = tf.einsum("vrai,vrai->vra", v0, v0)
        dot01 = tf.einsum("vrai,vrai->vra", v0, v1)
        dot02 = tf.einsum("vrai,vrai->vra", v0, v2)
        dot11 = tf.einsum("vrai,vrai->vra", v1, v1)
        dot12 = tf.einsum("vrai,vrai->vra", v1, v2)

        denominator = dot00 * dot11 - dot01 * dot01

        point_2_weight = (dot11 * dot02 - dot01 * dot12) / denominator
        point_1_weight = (dot00 * dot12 - dot01 * dot02) / denominator
        point_0_weight = 1 - point_2_weight - point_1_weight

        interpolation_weights = tf.stack([point_2_weight, point_1_weight, point_0_weight], axis=-1)

        # 5.) Get projection indices
        # 'projections_indices': (vertices, 3, n_radial, n_angular)
        projections_indices = tf.cast(tf.gather(neighborhoods_indices, closest_proj, batch_dims=1), tf.float32)

        # 6.) Return barycentric coordinates tensor
        # (vertices, n_radial, n_angular, 3, 2)
        return tf.stack([tf.transpose(projections_indices, perm=[0, 2, 3, 1]), interpolation_weights], axis=-1)
