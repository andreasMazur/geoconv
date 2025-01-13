from geoconv.preprocessing.barycentric_coordinates import create_template_matrix
from geoconv.tensorflow.utils.compute_shot_lrf import logarithmic_map, compute_distance_matrix, knn_shot_lrf

import tensorflow as tf
import numpy as np
import sys


@tf.function
def check_delaunay_condition(closet_proj, other_proj, interpolation_w_indices):
    # 'to_filter': (x, 5), whereby 'x' depends on how many 'inf'-values exist in 'interpolation_w_indices' and
    #                      5 comes from the length of: (n, n_radial, n_angular, n_neighbors - 1, n_neighbors - 1)
    #                      which is the shape of index vectors for 'interpolation_w_indices'
    to_filter = tf.where(interpolation_w_indices != np.inf)

    # Get triangles which include the template vertex
    # 'snd_idx'/'thr_idx': (x, 4), describing snd & third neighbor indices
    snd_idx = to_filter[:, :4]
    thr_idx = tf.concat([to_filter[:, :3], to_filter[:, 4:]], axis=-1)

    # 'snd_proj'/'thr_proj': (x, 2), describing 2D coordinates of projected neighbors
    snd_proj = tf.gather_nd(other_proj, snd_idx)
    thr_proj = tf.gather_nd(other_proj, thr_idx)

    # 'triangles': (x, 3, 2)
    triangles = tf.stack([tf.gather_nd(closet_proj[:, :, :, 0, :], to_filter[:, :3]), snd_proj, thr_proj], axis=1)

    # Compute mask to filter out elements of triangle from neighborhood.
    # The elements in the neighborhoods will be checked on whether they lie within the circumcircle of the triangle.
    # If any element in neighborhoods lies within circumcircle, discard the neighborhood. See determinant check below.
    # 'neighborhoods'/'trues': (x, n_neighbors - 1, 2)
    neighborhoods = tf.gather_nd(other_proj, to_filter[:, :3])
    trues = tf.ones_like(neighborhoods, dtype=tf.bool)

    # 'falses': (x, 2)
    falses = tf.zeros(shape=(tf.shape(snd_idx)[0], 2), dtype=tf.bool)

    # 'mask': (x, n_neighbors - 1, 2)
    mask = tf.logical_and(
        tf.tensor_scatter_nd_update(
            trues, tf.stack([tf.range(tf.shape(neighborhoods)[0], dtype=tf.int64), snd_idx[:, -1]], axis=-1), falses
        ),
        tf.tensor_scatter_nd_update(
            trues, tf.stack([tf.range(tf.shape(neighborhoods)[0], dtype=tf.int64), thr_idx[:, -1]], axis=-1), falses
        )
    )

    # 'neighborhoods': (x, n_neighbors - 1 - 2, 2)
    n_shape = tf.shape(neighborhoods)
    neighborhoods = tf.reshape(
        tf.boolean_mask(neighborhoods, tf.logical_and(mask[:, :, 0], mask[:, :, 1])),
        (n_shape[0], n_shape[1] - 2, n_shape[2])
    )

    # Determinant check to see whether neighboring vertices are within the circumcircle of a triangle
    # i)  Repeat triangle for each element in the neighborhood
    # ii) Perform determinant check for all neighbors and triangles
    # 'column_<NUMBER>': (x, n_neighbors - 1 - 2, 4)
    column_1 = tf.concat(
        [tf.tile(triangles[:, None, :, 0], (1, tf.shape(neighborhoods)[1], 1)), neighborhoods[:, :, :1]], axis=-1
    )
    column_2 = tf.concat(
        [tf.tile(triangles[:, None, :, 1], (1, tf.shape(neighborhoods)[1], 1)), neighborhoods[:, :, 1:]], axis=-1
    )
    column_3 = tf.math.square(column_1) + tf.math.square(column_2)
    column_4 = tf.ones_like(column_3)

    # Check determinant for each neighborhood
    # 'matrix = tf.stack([column_1, column_2, column_3, column_4], axis=-1)':
    #       (x, n_neighbors - 1 - 2, 4, 4)
    #
    # 'tf.linalg.det(matrix)': (x, n_neighbors - 1 - 2)

    # Remove any triangle which produces a circumcircle that includes other projections other than its own
    # 'condition': (x,)
    condition = tf.reduce_prod(
        tf.cast(tf.linalg.det(tf.stack([column_1, column_2, column_3, column_4], axis=-1)) <= 0, tf.int32), axis=-1
    )

    # 'to_filter' points to possible pairs that could construct a triangle
    condition = to_filter[tf.cast(condition, tf.bool)]

    # Overwrite all except
    return tf.tensor_scatter_nd_update(
        tf.ones_like(interpolation_w_indices) * np.inf, condition, tf.gather_nd(interpolation_w_indices, condition)
    )


@tf.function  # (jit_compile=True)
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

    ###########################################
    # 1) Compute distance to template vertices
    ###########################################
    projections = tf.expand_dims(tf.expand_dims(projections, axis=1), axis=1)

    # 'closest_idx_hierarchy': (vertices, n_radial, n_angular, n_neighbors, 2)
    closest_idx_hierarchy = tf.expand_dims(template, axis=2) - projections

    ##############################################################################################
    # 2) Retrieve neighborhood indices of closest projections (NOT equal to shape vertex indices)
    ##############################################################################################
    # 'closest_idx_hierarchy': (vertices, n_radial, n_angular, n_neighbors)
    closest_idx_hierarchy = tf.argsort(tf.linalg.norm(closest_idx_hierarchy, axis=-1), axis=-1)

    #################################################
    # 3) Determine 'closest' and 'other' projections
    #################################################
    # 'closet_proj': (vertices, n_radial, n_angular, 1, 2)
    closet_proj = tf.gather(tf.squeeze(projections), closest_idx_hierarchy[:, :, :, 0], batch_dims=1)[:, :, :, None, :]
    # 'other_proj':  (vertices, n_radial, n_angular, n_neighbors - 1, 2)
    other_proj = tf.gather(tf.squeeze(projections), closest_idx_hierarchy[:, :, :, 1:], batch_dims=1)

    #####################################
    # 4) Compute barycentric coordinates
    #####################################
    v0 = other_proj - closet_proj
    v1 = other_proj - closet_proj
    v2 = tf.expand_dims(template, axis=-2) - closet_proj

    dot00 = tf.einsum("vrani,vrani->vran", v0, v0)
    dot01 = tf.einsum("vrani,vrami->vranm", v0, v1)  # dot01[..., n, m] = dot product of neighbor n with neighbor m
    dot02 = tf.einsum("vrani,vrai->vran", v0, tf.squeeze(v2))

    dot11 = tf.einsum("vrani,vrani->vran", v1, v1)
    dot12 = tf.einsum("vrani,vrai->vran", v1, tf.squeeze(v2))

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

    # 'interpolation_weights': (vertices, radial, angular, n_neighbors - 1, n_neighbors - 1, 3)
    interpolation_weights = tf.stack([point_0_weight, point_2_weight, point_1_weight], axis=-1)

    # Set negative-, zero- and NAN-interpolation values to infinity
    to_filter = tf.where(tf.logical_or(interpolation_weights <= 0., tf.math.is_nan(interpolation_weights)))
    interpolation_weights = tf.tensor_scatter_nd_update(
        interpolation_weights, to_filter, tf.cast(tf.fill((tf.shape(to_filter)[0],), np.inf), tf.float64)
    )

    # Encourage using BC with smallest inf-norm
    interpolation_w_indices = tf.linalg.norm(tf.math.square(interpolation_weights), axis=-1, ord=np.inf)

    # Check Delaunay condition
    interpolation_w_indices = check_delaunay_condition(closet_proj, other_proj, interpolation_w_indices)

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

    # Replace infinity interpolation coefficients with zeros to prevent any contribution of this template vertex
    to_filter = tf.where(interpolation_weights == np.inf)[:, :3]
    interpolation_weights = tf.tensor_scatter_nd_update(
        interpolation_weights,
        to_filter,
        tf.cast(tf.tile([[0., 0., 0.]], multiples=[tf.shape(to_filter)[0], 1]), tf.float64)
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
                distance_matrix = compute_distance_matrix(tf.cast(vertices[0], tf.float32))
                radii = tf.gather(
                    distance_matrix, tf.argsort(distance_matrix, axis=-1)[:, n_neighbors], batch_dims=1
                )
                avg_radius = avg_radius + tf.reduce_sum(radii)
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
