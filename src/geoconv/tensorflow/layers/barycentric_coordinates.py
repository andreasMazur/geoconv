from geoconv.preprocessing.barycentric_coordinates import create_template_matrix
from geoconv.tensorflow.utils.compute_shot_lrf import logarithmic_map, compute_distance_matrix, knn_shot_lrf

import tensorflow as tf
import numpy as np
import sys


@tf.function(jit_compile=True)
def unravel_square_matrix_index(batched_idx_matrices, dim):
    row_indices = tf.floor(batched_idx_matrices / tf.cast(dim, tf.int64))
    col_indices = tf.cast(batched_idx_matrices, tf.float64) - row_indices * tf.cast(dim, tf.float64)
    return tf.stack([tf.cast(row_indices, tf.int64), tf.cast(col_indices, tf.int64)], axis=-1)


@tf.function(jit_compile=True)
def pairwise_bc(closet_proj, other_proj, template):
    v0_v1 = other_proj - closet_proj
    v2 = tf.expand_dims(template, axis=-2) - closet_proj

    dot00 = tf.einsum("vrani,vrani->vran", v0_v1, v0_v1)
    # dot01[..., n, m] = dot product of neighbor n with neighbor m
    dot01 = tf.einsum("vrani,vrami->vranm", v0_v1, v0_v1)
    dot02 = tf.einsum("vrani,vrai->vran", v0_v1, tf.squeeze(v2))

    dot11 = tf.einsum("vrani,vrani->vran", v0_v1, v0_v1)
    dot12 = tf.einsum("vrani,vrai->vran", v0_v1, tf.squeeze(v2))

    denominator = tf.einsum("vran,vram->vranm", dot00, dot11) - dot01 * dot01
    # Set diagonal elements to be filtered out
    denominator = tf.linalg.set_diag(tf.cast(denominator, tf.float32), tf.zeros(tf.shape(denominator)[:4]))
    denominator = tf.cast(denominator, tf.float64)
    denominator = 1 / denominator  # NAN-values are filtered later. Keep this to make shapes fit for EINSUM.

    point_2_weight = tf.einsum("vram,vran->vranm", dot11, dot02) - tf.einsum("vranm,vram->vranm", dot01, dot12)
    point_2_weight = point_2_weight * denominator

    point_1_weight = tf.einsum("vran,vram->vranm", dot00, dot12) - tf.einsum("vranm,vran->vranm", dot01, dot02)
    point_1_weight = point_1_weight * denominator

    point_0_weight = 1 - point_2_weight - point_1_weight

    # (vertices, radial, angular, n_neighbors - 1, n_neighbors - 1, 3)
    return tf.stack([point_0_weight, point_2_weight, point_1_weight], axis=-1)


@tf.function(jit_compile=True)
def compute_interpolation_weights(template, projections):
    # 0.) Compute distance to projections - 'distances': (n_vertices, n_radial, n_angular, n_neighbors)
    distances = template[None, :, :, None, :] - projections[:, None, None, :, :]
    distances = tf.linalg.norm(distances, axis=-1)
    closest_idx_hierarchy = tf.argsort(distances, axis=-1)

    # 1.) Remove the closest projection from other projections
    # 'p_closest_idx': (n_vertices, n_radial, n_angular)
    p_closest_idx = closest_idx_hierarchy[:, :, :, 0]
    # 'p_other_indices': (n_vertices, n_radial, n_angular, n_neighbors - 1)
    p_other_indices = closest_idx_hierarchy[:, :, :, 1:]

    # 'p_closest': (n_vertices, n_radial, n_angular, 2)
    p_closest = tf.gather(projections, p_closest_idx, batch_dims=1)

    # Repeat projections for each template vertex - 'projections': (n_vertices, n_radial, n_angular, n_neighbors, 2)
    template_shape = tf.shape(template)
    projections = tf.tile(projections[:, None, None, :, :], (1, template_shape[0], template_shape[1], 1, 1))
    p_shape = tf.shape(projections)

    # 'other_projections':  (n_vertices, n_radial, n_angular, n_neighbors - 1, 2)
    other_projections = tf.gather(projections, p_other_indices, batch_dims=3)

    # 2.) Find all triangles with 'v_closest' that include 'template_vertex'
    # Compute barycentric coordinates - 'bc': (n_vertices, n_radial, n_angular, n_neighbors - 1, n_neighbors - 1, 3)
    bc = pairwise_bc(p_closest[:, :, :, None, :], other_projections, template)

    # Compute total template-to-triangle-vertex-distance
    # 'closest_distances': (n_vertices, n_radial, n_angular, 1)
    # 'other_distances': (n_vertices, n_radial, n_angular, n_neighbors - 1)
    closest_distances = tf.reduce_min(distances, axis=-1)[..., None]
    other_distances = tf.gather(distances, p_other_indices, batch_dims=3)

    # Compute total distance and variance among distances from the two remaining triangle vertices to the template
    # vertex, assuming closest projection is part of triangle
    # 'paired_distances': (n_vertices, n_radial, n_angular, n_neighbors - 1, n_neighbors - 1, 2)
    paired_distances = tf.stack(
        [
            tf.tile(other_distances[..., None], (1, 1, 1, 1, p_shape[-2] - 1)),
            tf.tile(other_distances[..., None, :], (1, 1, 1, p_shape[-2] - 1, 1))
        ], axis=-1
    )
    # 'total_distance': (n_vertices, n_radial, n_angular, n_neighbors - 1, n_neighbors - 1)
    total_distance = tf.reduce_sum(paired_distances, axis=-1) + closest_distances[..., None]

    # 'variance': (n_vertices, n_radial, n_angular, n_neighbors - 1, n_neighbors - 1)
    variance = tf.math.reduce_variance(
        tf.concat(
            [
                paired_distances,
                tf.tile(closest_distances[..., None, None, None, 0], (1, 1, 1, p_shape[-2] - 1, p_shape[-2] - 1, 1))
            ], axis=-1
        ), axis=-1
    )

    # Set distance of pairs to infinity, if the bc of the triangle-vertex-pairs indicate non-fitting triangle
    to_filter = tf.where(tf.logical_or(tf.logical_or(bc < 0., bc > 1.), tf.math.is_nan(bc)))
    total_distance = total_distance / tf.reduce_max(total_distance)
    total_distance = tf.tensor_scatter_nd_update(
        total_distance, to_filter[:, :5], tf.cast(tf.fill((tf.shape(to_filter)[0],), np.inf), tf.float64)
    )

    # Compute indices of the pair that is closest to the template vertex
    # 'total_distance'/'variance': (n_vertices * n_radial * n_angular, (n_neighbors - 1) ** 2)
    total_distance = tf.reshape(total_distance, (-1, (p_shape[-2] - 1) * (p_shape[-2] - 1)))

    # 'indices' are indices w.r.t. projection neighborhood without closest neighbor to template vertex
    # 'indices': (n_vertices * n_radial * n_angular, 2)
    indices = unravel_square_matrix_index(tf.argmin(total_distance, axis=-1), p_shape[-2] - 1)

    # 'indices': (n_vertices, n_radial, n_angular, 2)
    indices = tf.reshape(indices, (p_shape[0], p_shape[1], p_shape[2], 2))

    # triangles = tf.concat([p_closest[..., None], tf.gather(projections, indices, batch_dims=3)], axis=-1)
    # 'bc': (n_vertices, n_radial, n_angular, 3)
    bc = tf.gather_nd(bc, indices, batch_dims=3)

    # Get triangle indices of two remaining vertices (w.r.t. neighborhood indices - NOT global shape-vertex indices)
    # (n_vertices, n_radial, n_angular, 2)
    p_other_indices = tf.gather(p_other_indices, indices, batch_dims=3)

    # Return bc and the corresponding indices
    return bc, tf.concat([p_closest_idx[..., None], p_other_indices], axis=-1)


@tf.function(jit_compile=True)
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
    template = tf.cast(template, tf.float64)
    projections = tf.cast(projections, tf.float64)
    interpolation_weights, interpolation_indices = compute_interpolation_weights(template, projections)

    # Replace 'nan'-interpolation coefficients with zeros to prevent any contribution of this template vertex
    to_filter = tf.where(
        tf.logical_or(tf.math.is_nan(interpolation_weights), tf.math.is_inf(interpolation_weights))
    )[:, :3]
    zeros = tf.cast(tf.tile([[0., 0., 0.]], multiples=[tf.shape(to_filter)[0], 1]), tf.float64)
    interpolation_weights = tf.tensor_scatter_nd_update(interpolation_weights, to_filter, zeros)
    interpolation_indices = tf.tensor_scatter_nd_update(interpolation_indices, to_filter, tf.cast(zeros, tf.int32))

    return interpolation_weights, interpolation_indices


class BarycentricCoordinates(tf.keras.layers.Layer):
    """A parameter-free neural network layer that approximates barycentric coordinates (BC).

    Attributes
    ----------
    n_radial: int
        The amount of radial coordinates of the template for which BC shall be computed.
    n_angular: int
        The amount of angular coordinates of the template for which BC shall be computed.
    """
    def __init__(self, n_radial, n_angular, neighbors_for_lrf=16):
        super().__init__()
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.template = None
        self.neighbors_for_lrf = neighbors_for_lrf

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
        if template_radius is None:
            assert data is not None, "If 'template_radius' is not given, you must provide 'data'."
            assert n_neighbors is not None, "If 'template_radius' is not given, you must provide 'n_neighbors'."
            assert template_scale is not None, "If 'template_radius' is not given, you must provide 'template_scale'."

        # If no template radius is given, compute the template radius
        if template_radius is None:
            avg_radius, vertices_count = 0, 0
            for idx, (vertices, _) in enumerate(data):
                sys.stdout.write(f"\rCurrently at point-cloud {idx}.")
                # 1.) Get local reference frames
                # 'lrfs': (vertices, 3, 3)
                lrfs, neighborhoods, neighborhoods_indices = knn_shot_lrf(self.neighbors_for_lrf, vertices[0])

                # 2.) Project neighborhoods into their lrfs using the logarithmic map
                # 'projections': (vertices, n_neighbors, 2)
                projections = logarithmic_map(lrfs, neighborhoods)

                # 3.) Use length of farthest projection as radius
                radii = tf.reduce_max(tf.linalg.norm(projections, axis=-1), axis=-1)

                # 4.) Add all radii
                avg_radius = avg_radius + tf.reduce_sum(radii)

                # 5.) Remember amount of collected radii for averaging
                vertices_count = vertices_count + tf.cast(tf.shape(radii)[0], tf.float32)
            avg_radius = avg_radius / vertices_count
            template_radius = avg_radius * template_scale

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
        self.call_helper(vertices[0])
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
        # 1.) Get local reference frames
        # 'lrfs': (vertices, 3, 3)
        lrfs, neighborhoods, neighborhoods_indices = knn_shot_lrf(self.neighbors_for_lrf, vertices)

        # 2.) Project neighborhoods into their lrfs using the logarithmic map
        # 'projections': (vertices, n_neighbors, 2)
        projections = logarithmic_map(lrfs, neighborhoods)

        # 3.) Compute barycentric coordinates
        # 'interpolation_weights': (vertices, n_radial, n_angular, 3)
        # 'closest_proj': (vertices, n_radial, n_angular, 3)
        interpolation_weights, closest_proj = compute_bc(self.template, projections)
        interpolation_weights = tf.cast(interpolation_weights, tf.float32)

        # 4.) Get projection indices (convert neighborhood indices to shape vertex indices)
        # 'projections_indices': (vertices, n_radial, n_angular, 3)
        projections_indices = tf.cast(tf.gather(neighborhoods_indices, closest_proj, batch_dims=1), tf.float32)

        # 5.) Return barycentric coordinates tensor
        # (vertices, n_radial, n_angular, 3, 2)
        return tf.stack([projections_indices, interpolation_weights], axis=-1)
