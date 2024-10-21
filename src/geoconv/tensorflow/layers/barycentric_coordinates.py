from geoconv.preprocessing.barycentric_coordinates import create_template_matrix
from geoconv.tensorflow.utils.compute_shot_lrf import (
    group_neighborhoods, shot_lrf, logarithmic_map, compute_distance_matrix
)

import tensorflow as tf
import numpy as np


@tf.function
def compute_bc(template, projections):
    """Computes barycentric coordinates for a given template in given projections.

    Parameters
    ----------
    template: tf.Tensor
        A 3D-tensor of shape (n_radial, n_angular, 2) that contains 2D cartesian coordinates for template vertices.
    projections: tf.Tensor
        A 3D-tensor of shape (vertices, n_neighbors, 2) that contains all projected neighborhoods in 2D cartesian
        coordinates. I.e., 'projections[i, j]' contains 2D coordinates of vertex 'j' in neighborhood 'i'.

    Returns
    -------
    (tf.Tensor, tf.Tensor):
        A 4D-tensor of shape (vertices, n_radial, n_angular, 3) that contains barycentric coordinates, i.e.,
        interpolation coefficients, for all template vertices within each projected neighborhood. Additionally,
        another 4D-tensor of shape (vertices, n_radial, n_angular, 3) that contains the vertex indices of the closest
        projected vertices to the template vertices in each neighborhood.
    """
    # 1) Compute distance to template vertices
    projections = tf.expand_dims(tf.expand_dims(projections, axis=1), axis=1)

    # 'closest_idx_hierarchy': (vertices, n_radial, n_angular, n_neighbors, 2)
    closest_idx_hierarchy = tf.expand_dims(template, axis=2) - projections

    # 2) Retrieve neighborhood indices of two closest projections (NOT equal to shape vertex indices)
    # 'closest_idx_hierarchy': (vertices, n_radial, n_angular, n_neighbors)
    closest_idx_hierarchy = tf.argsort(tf.linalg.norm(closest_idx_hierarchy, axis=-1), axis=-1)

    # 3) Determine 'closest' and 'other' projections
    # 'closet_proj': (vertices, n_radial, n_angular, 1, 2)
    closet_proj = tf.gather(tf.squeeze(projections), closest_idx_hierarchy[:, :, :, 0], batch_dims=1)[:, :, :, None, :]
    # 'other_proj':  (vertices, n_radial, n_angular, n_neighbors - 1, 2)
    other_proj = tf.gather(tf.squeeze(projections), closest_idx_hierarchy[:, :, :, 1:], batch_dims=1)

    # 4) Compute barycentric coordinates
    v0 = other_proj - closet_proj
    v1 = other_proj - closet_proj
    v2 = tf.expand_dims(template, axis=-2) - closet_proj

    dot00 = tf.einsum("vrani,vrani->vran", v0, v0)
    dot01 = tf.einsum("vrani,vrami->vranm", v0, v1)  # dot01[..., n, m] = dot product of neighbor n with neighbor m
    dot02 = tf.einsum("vrani,vrai->vran", v0, tf.squeeze(v2))

    dot11 = tf.einsum("vrani,vrani->vran", v1, v1)
    dot12 = tf.einsum("vrani,vrai->vran", v1, tf.squeeze(v2))

    denominator = tf.einsum("vran,vram->vranm", dot00, dot11) - dot01 * dot01

    # Avoid dividing by zero
    zero_indices = tf.where(denominator == 0.)
    denominator = tf.tensor_scatter_nd_update(denominator, zero_indices, tf.fill((tf.shape(zero_indices)[0],), 1e-10))

    point_2_weight = tf.einsum("vram,vran->vranm", dot11, dot02) - tf.einsum("vranm,vram->vranm", dot01, dot12)
    point_2_weight = point_2_weight / denominator

    point_1_weight = tf.einsum("vran,vram->vranm", dot00, dot12) - tf.einsum("vranm,vran->vranm", dot01, dot02)
    point_1_weight = point_1_weight / denominator

    point_0_weight = 1 - point_2_weight - point_1_weight

    # 'interpolation_weights': (vertices, radial, angular, n_neighbors - 1, n_neighbors - 1, 3)
    interpolation_weights = tf.stack([point_0_weight, point_2_weight, point_1_weight], axis=-1)

    # Set negative and zero interpolation values to infinity
    to_filter = tf.where(interpolation_weights <= 0.)
    interpolation_weights = tf.tensor_scatter_nd_update(
        interpolation_weights, to_filter, tf.fill((tf.shape(to_filter)[0],), np.inf)
    )

    # Encourage using BC with smallest inf-norm
    interpolation_w_indices = tf.linalg.norm(tf.math.square(interpolation_weights), axis=-1, ord=np.inf)

    # From all possible interpolation weights, select BC with smallest sup-norm
    s = tf.shape(interpolation_w_indices)
    interpolation_w_indices = tf.reshape(interpolation_w_indices, (s[0], s[1], s[2], s[3] * s[4]))
    interpolation_w_indices = tf.argmin(interpolation_w_indices, axis=-1)

    # Recompute integer into (row, column)-tuple, so we can select from 'interpolation_weights'
    row_indices = tf.floor(interpolation_w_indices / tf.cast(s[3], tf.int64))
    col_indices = tf.cast(interpolation_w_indices, tf.float64) - row_indices * tf.cast(s[3], tf.float64)
    interpolation_w_indices = tf.stack([tf.cast(row_indices, tf.int64), tf.cast(col_indices, tf.int64)], axis=-1)

    # Gather corresponding BC using the found tuples
    interpolation_weights = tf.gather_nd(interpolation_weights, interpolation_w_indices, batch_dims=3)

    # Replace infinity interpolation coefficients with zeros to cancel any contribution of this template vertex
    to_filter = tf.where(interpolation_weights == np.inf)[:, :3]
    interpolation_weights = tf.tensor_scatter_nd_update(
        interpolation_weights,
        to_filter,
        tf.tile([[0., 0., 0.]], multiples=[tf.shape(to_filter)[0], 1])
    )

    # Convert BC-indices
    interpolation_w_indices = tf.gather(closest_idx_hierarchy[:, :, :, 1:], interpolation_w_indices, batch_dims=3)

    # Group all associated BC-indices
    interpolation_w_indices = tf.concat(
        [tf.cast(closest_idx_hierarchy[:, :, :, 0, None], tf.int32), interpolation_w_indices], axis=-1
    )

    return interpolation_weights, interpolation_w_indices


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
    def __init__(self, n_radial, n_angular, n_neighbors, template_scale=0.75, neighbor_limit=20):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.neighbor_limit = neighbor_limit
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

        # If no template radius is given, compute the template radius
        if template_radius is None:
            avg_radius, vertices_count = 0, 0
            for idx, (vertices, _) in enumerate(data):
                distance_matrix = compute_distance_matrix(tf.cast(vertices[0], tf.float32))
                radii = tf.gather(
                    distance_matrix, tf.argsort(distance_matrix, axis=-1)[:, self.n_neighbors], batch_dims=1
                )
                avg_radius = avg_radius + tf.reduce_sum(radii)
                vertices_count = vertices_count + tf.cast(tf.shape(radii)[0], tf.float32)
            avg_radius = avg_radius / vertices_count
            template_radius = avg_radius * self.template_scale

        # Initialize template
        self.template = tf.constant(
            create_template_matrix(
                n_radial=self.n_radial, n_angular=self.n_angular, radius=template_radius, in_cart=True
            ), dtype=tf.float32
        )

        # Return used template radius
        return template_radius

    @tf.function(jit_compile=True)
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

    @tf.function(jit_compile=True)
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
        # 1.) Compute radius for local parameterization spaces. Keep it equal for all for comparability.
        # 'distance_matrix': (vertices, vertices)
        # 'radius': ()
        distance_matrix = compute_distance_matrix(vertices)
        radius = tf.reduce_mean(
            tf.gather(distance_matrix, tf.argsort(distance_matrix, axis=-1)[:, self.n_neighbors], batch_dims=1)
        )

        # 2.) Get vertex-neighborhoods
        # 'neighborhoods': (vertices, n_neighbors, 3)
        neighborhoods, neighborhoods_indices = group_neighborhoods(
            vertices, radius, neighbor_limit=self.neighbor_limit, distance_matrix=distance_matrix
        )

        # 3.) Get local reference frames
        # 'lrfs': (vertices, 3, 3)
        lrfs = shot_lrf(neighborhoods, radius)

        # 4.) Project neighborhoods into their lrfs using the logarithmic map
        # 'projections': (vertices, n_neighbors, 2)
        projections = logarithmic_map(lrfs, neighborhoods)

        # 5.) Compute barycentric coordinates
        # 'interpolation_weights': (vertices, n_radial, n_angular, 3)
        # 'closest_proj': (vertices, n_radial, n_angular, 3)
        interpolation_weights, closest_proj = compute_bc(self.template, projections)

        # 6.) Get projection indices (convert neighborhood indices to shape vertex indices)
        # 'projections_indices': (vertices, n_radial, n_angular, 3)
        projections_indices = tf.cast(tf.gather(neighborhoods_indices, closest_proj, batch_dims=1), tf.float32)

        # 7.) Return barycentric coordinates tensor
        # (vertices, n_radial, n_angular, 3, 2)
        return tf.stack([projections_indices, interpolation_weights], axis=-1)
