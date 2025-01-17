from geoconv.preprocessing.barycentric_coordinates import create_template_matrix
from geoconv.tensorflow.utils.compute_shot_lrf import logarithmic_map, knn_shot_lrf

import tensorflow as tf
import numpy as np
import sys


@tf.function(jit_compile=True)
def compute_det(batched_matrices):
    a = batched_matrices[..., 0, 0]
    b = batched_matrices[..., 0, 1]
    c = batched_matrices[..., 0, 2]
    d = batched_matrices[..., 1, 0]
    e = batched_matrices[..., 1, 1]
    f = batched_matrices[..., 1, 2]
    g = batched_matrices[..., 2, 0]
    h = batched_matrices[..., 2, 1]
    i = batched_matrices[..., 2, 2]

    return a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h


@tf.function(jit_compile=True)
def delaunay_condition_check(triangles, projections):
    # 'column_1_2': (n_vertices, n_neighbors, `n_neighbors over 3`, 3, 2)
    # 'column_1_2[..., 0]' -> x-coordinate difference to triangle vertices from selected neighbor
    # 'column_1_2[..., 1]' -> y-coordinate difference to triangle vertices from selected neighbor
    column_1_2 = triangles[:, None, ...] - projections[..., None, None, :]

    # 'delaunay_check_matrix': (n_vertices, n_neighbors, `n_neighbors over 3`)
    # True if projection 'i' outside of circumcircle of triangle 'j' in neighborhood 'k'
    delaunay_check_matrix = tf.cast(
        compute_det(
            tf.stack(
                [
                    column_1_2[..., 0],
                    column_1_2[..., 1],
                    tf.math.square(column_1_2[..., 0]) + tf.math.square(column_1_2[..., 1])
                ], axis=-1  # last dim stacks columns of matrices
            )
        ) > 0., tf.int32
    )

    # 'delaunay_check_matrix': (n_vertices, `n_neighbors over 3`)
    # Every triangle should have 3 vertices that fall into their circumcircle (their own vertices)
    # Use '<=' to account for numerical inaccuracies
    delaunay_check_matrix = tf.reduce_sum(delaunay_check_matrix, axis=1) <= 3

    return delaunay_check_matrix


@tf.function(jit_compile=True)
def create_all_triangles(projections):
    p_shape = tf.shape(projections)

    # Create all possible (i, j, k) index triplets
    I, J, K = tf.meshgrid(tf.range(p_shape[-2]), tf.range(p_shape[-2]), tf.range(p_shape[-2]), indexing="ij")

    # Filter out invalid combinations (where i < j < k)
    triangle_indices = tf.where(tf.logical_and((I < J), (J < K)))

    # 'index_tensor': (n_vertices, `n_neighbors over 3`, 3, 2)
    triangles = tf.gather(projections, tf.tile(triangle_indices[None, ...], (p_shape[0], 1, 1)), batch_dims=1)

    return triangles, triangle_indices


@tf.function(jit_compile=True)
def compute_interpolation_coefficients(triangles, template):
    v0 = triangles[..., 2, :] - triangles[..., 0, :]
    v1 = triangles[..., 1, :] - triangles[..., 0, :]
    v2 = template[None, ..., None, :] - triangles[:, None, None, :, 0, :]

    dot00 = tf.einsum("ijk,ijk->ij", v0, v0)[:, None, None, :]
    dot01 = tf.einsum("ijk,ijk->ij", v0, v1)[:, None, None, :]
    dot02 = tf.einsum("ijk,irajk->iraj", v0, v2)
    dot11 = tf.einsum("ijk,ijk->ij", v1, v1)[:, None, None, :]
    dot12 = tf.einsum("ijk,irajk->iraj", v1, v2)

    denominator = 1 / (dot00 * dot11 - dot01 * dot01)
    point_2_weight = (dot11 * dot02 - dot01 * dot12) * denominator
    point_1_weight = (dot00 * dot12 - dot01 * dot02) * denominator
    point_0_weight = 1 - point_2_weight - point_1_weight

    return tf.stack([point_0_weight, point_1_weight, point_2_weight], axis=-1)


@tf.function(jit_compile=True)
def compute_interpolation_weights(template, projections):
    # 'triangles': (n_vertices, `n_neighbors over 3`, 3, 2)
    # 'triangle_indices': (`n_neighbors over 3`, 3)
    triangles, triangle_indices = create_all_triangles(projections)

    # 'delaunay_condition[x, y] = True' if 'triangles[x, y]' meets Delaunay condition
    # 'delaunay_condition': (n_vertices, `n_neighbors over 3`)
    delaunay_condition = delaunay_condition_check(triangles, projections)

    # 'barycentric_coordinates': (n_vertices, n_radial, n_angular, `n_neighbors over 3`, 3)
    barycentric_coordinates = compute_interpolation_coefficients(triangles, template)

    # 'mask': (n_vertices, n_radial, n_angular, `n_neighbors over 3`)
    bc_condition = tf.math.reduce_any(
        tf.logical_or(barycentric_coordinates > 1., barycentric_coordinates < 0.), axis=-1
    )
    mask = tf.logical_or(tf.logical_not(delaunay_condition[:, None, None, :]), bc_condition)

    # 'tri_distances': (n_vertices, n_radial, n_angular, `n_neighbors over 3`)
    tri_distances = tf.reduce_sum(
        tf.linalg.norm(triangles[:, None, None, ...] - template[None, :, :, None, None, :], axis=-1), axis=-1
    )

    # Set triangle distances to infinity where conditions aren't met
    mask_indices = tf.where(mask)
    tri_distances = tf.tensor_scatter_nd_update(
        tri_distances, mask_indices, tf.cast(tf.fill((tf.shape(mask_indices)[0],), np.inf), tf.float64)
    )
    closest_triangles = tf.argmin(tri_distances, axis=-1)

    # Select bc of closest possible triangle
    selected_bc = tf.gather(barycentric_coordinates, closest_triangles, batch_dims=3)
    selected_indices = tf.cast(tf.gather(triangle_indices, closest_triangles), tf.int32)

    # Might happen that no triangles fit for a template vertex. Set those interpolation coefficients to zero.
    correction_mask = tf.where(tf.reduce_all(mask, axis=-1))
    zeros = tf.cast(tf.zeros((tf.shape(correction_mask)[0], 3)), tf.float64)
    selected_bc = tf.tensor_scatter_nd_update(selected_bc, correction_mask, zeros)
    selected_indices = tf.tensor_scatter_nd_update(selected_indices, correction_mask, tf.cast(zeros, tf.int32))

    return selected_bc, selected_indices


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
                lrfs, neighborhoods, neighborhoods_indices = knn_shot_lrf(n_neighbors, vertices[0])

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
