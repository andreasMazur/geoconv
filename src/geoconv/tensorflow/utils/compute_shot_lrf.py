import tensorflow as tf
import numpy as np


@tf.function(jit_compile=True)
def compute_distance_matrix(vertices):
    """Computes the Euclidean distance between given vertices.

    Parameters
    ----------
    vertices: tf.Tensor
        The vertices to compute the distance between.

    Returns
    -------
    tf.Tensor:
        A square distance matrix for the given vertices.
    """
    vertices = tf.cast(vertices, tf.float64)
    
    norm = tf.einsum("ij,ij->i", vertices, vertices)
    norm = tf.reshape(norm, (-1, 1)) - 2 * tf.einsum("ik,jk->ij", vertices, vertices) + tf.reshape(norm, (1, -1))

    where_nans = tf.where(tf.math.is_nan(tf.sqrt(norm)))
    norm = tf.tensor_scatter_nd_update(
        norm, where_nans, tf.zeros(shape=(tf.shape(where_nans)[0],), dtype=tf.float64)
    )

    return tf.cast(tf.sqrt(norm), tf.float32)


@tf.function(jit_compile=True)
def disambiguate_axes(neighborhood_vertices, eigen_vectors):
    """Disambiguate axes returned by local Eigenvalue analysis.

    Disambiguation follows the formal procedure as described in:
    > [SHOT: Unique signatures of histograms for surface and texture
     description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    neighborhood_vertices: tf.Tensor
        The vertices of the neighborhoods.
    eigen_vectors: tf.Tensor
        The Eigenvectors of all neighborhoods for one dimension, i.e. it has size (#neighborhoods, 3).
        E.g. the x-axes.

    Returns
    -------
    tf.Tensor:
        The disambiguated Eigenvectors.
    """
    neg_eigen_vectors = -eigen_vectors
    ev_count = tf.math.count_nonzero(tf.einsum("nvk,nk->nv", neighborhood_vertices, eigen_vectors) >= 0, axis=-1)
    ev_neg_count = tf.math.count_nonzero(tf.einsum("nvk,nk->nv", neighborhood_vertices, -eigen_vectors) > 0., axis=-1)
    return tf.gather(
        tf.stack([neg_eigen_vectors, eigen_vectors], axis=1), tf.cast(ev_count >= ev_neg_count, tf.int32), batch_dims=1
    )


@tf.function(jit_compile=True)
def shot_lrf(neighborhoods, radii):
    """Computes SHOT local reference frames.

    SHOT computation was introduced in:
    > [SHOT: Unique signatures of histograms for surface and texture
     description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    neighborhoods: tf.Tensor
        The vertices of the neighborhoods shifted around the neighborhood origin.
    radii: tf.Tensor
        A 1D-tensor containing the radii of each neighborhood. I.e., its first dimension needs to be of the same size
        as the first dimension of the 'neighborhoods'-tensor.

    Returns
    -------
    tf.Tensor:
        Local reference frames for all given neighborhoods.
    """
    # 1.) Compute Eigenvectors
    # Calculate neighbor weights
    # 'distance_weights': (vertices, n_neighbors)
    distance_weights = tf.expand_dims(radii, axis=-1) - tf.linalg.norm(neighborhoods, axis=-1)

    # Compute weighted covariance matrices
    # 'weighted_cov': (vertices, 3, 3)
    weighted_cov = tf.einsum("nv,nvi,nvj->nij", distance_weights, neighborhoods, neighborhoods)

    # 2.) Disambiguate axes
    # First eigen vector corresponds to smallest eigen value (i.e. plane normal)
    _, eigen_vectors = tf.linalg.eigh(weighted_cov)

    # Columns contain eigenvectors
    x_axes = disambiguate_axes(neighborhoods, eigen_vectors[:, :, 2])
    z_axes = disambiguate_axes(neighborhoods, eigen_vectors[:, :, 0])
    y_axes = tf.linalg.cross(z_axes, x_axes)

    return tf.stack([z_axes, y_axes, x_axes], axis=-1)


# @tf.function(jit_compile=True)
def logarithmic_map(lrfs, neighborhoods):
    """Computes projections of neighborhoods into their local reference frames.

    Parameters
    ----------
    lrfs: tf.Tensor
        A 3D-tensor of shape (vertices, 3, 3) that contains the axes of local reference frames.
    neighborhoods: tf.Tensor
        A 3D-tensor of shape (vertices, n_neighbors, 3) that contains the neighborhoods around all vertices.

    Returns
    -------
    tf.Tensor:
        A 3D-tensor of shape (vertices, n_neighbors, 2) that contains the coordinates of the neighbor-projections
        within the tangent plane. Euclidean distance are preserved and used as an approximate to geodesic distances.
    """
    # Get tangent plane normals (z-axes of lrfs)
    normals = lrfs[:, 0, :]

    # Compute tangent plane projections (logarithmic map)
    scaled_normals = neighborhoods @ tf.expand_dims(normals, axis=-1) * tf.expand_dims(normals, axis=1)
    projections = neighborhoods - scaled_normals

    # Basis change of neighborhoods into lrf coordinates
    projections = tf.einsum("vij,vnj->vni", tf.linalg.inv(tf.transpose(lrfs, perm=[0, 2, 1])), projections)[:, :, 1:]

    # Use 'projection / adjacent * hypotenuse' as estimate to geodesic distance
    adj, hy = tf.linalg.norm(projections, axis=-1), tf.linalg.norm(neighborhoods, axis=-1)
    zero_indices = tf.where(adj == 0.)
    adj = tf.tensor_scatter_nd_update(adj, zero_indices, tf.ones((tf.shape(zero_indices)[0],)))
    hy = tf.tensor_scatter_nd_update(hy, zero_indices, tf.ones((tf.shape(zero_indices)[0],)))

    # Rescale projections to their original Euclidean distances
    projections = projections / adj[..., None] * hy[..., None]

    return projections


@tf.function(jit_compile=True)
def knn_shot_lrf(k_neighbors, vertices, repetitions=4):
    # 1.) Compute radius for local parameterization spaces. Keep it equal for all for comparability.
    # 'distance_matrix': (vertices, vertices)
    # 'radii': (vertices,)
    distance_matrix = compute_distance_matrix(vertices)
    radii = tf.gather(distance_matrix, tf.argsort(distance_matrix, axis=-1)[:, k_neighbors], batch_dims=1)

    # 2.) Get vertex-neighborhoods
    # 'neighborhoods': (vertices, n_neighbors, 3)
    neighborhoods, neighborhood_indices = tf.math.top_k(-distance_matrix, k_neighbors)
    neighborhoods = tf.gather(vertices, neighborhood_indices, axis=0) - tf.expand_dims(vertices, axis=1)

    # 3.) Get local reference frames
    # 'lrfs': (vertices, 3, 3)
    lrfs = shot_lrf(neighborhoods, radii)

    # 4.) Make normal vectors point away from centroid (outwards from shape)
    signs = -tf.cast(tf.einsum("vi,vi->v", lrfs[:, :, 0], tf.reduce_mean(vertices, axis=0) - vertices) >= 0, tf.int32)
    signs = signs + tf.cast(signs == 0, tf.int32)
    normals = tf.expand_dims(tf.cast(signs, tf.float32), axis=-1) * lrfs[:, :, 0]
    lrfs = tf.stack([normals, lrfs[:, :, 1], lrfs[:, :, 2]], axis=-1)

    # 5.) Make normal vectors in neighborhoods point the same direction
    # (non-convex shapes -> "outwards" might differ locally)
    for rep in range(repetitions):
        normals = tf.gather(lrfs[:, :, 0], neighborhood_indices[:, 1:])
        signs = -tf.cast(
            tf.reduce_sum(
                tf.cast(tf.einsum("vi,vni->vn", lrfs[:, :, 0], normals) >= 0, tf.int32), axis=-1
            ) <= tf.math.floordiv(k_neighbors, 2),
            tf.int32
        )
        signs = signs + tf.cast(signs == 0, tf.int32)
        normals = tf.expand_dims(tf.cast(signs, tf.float32), axis=-1) * lrfs[:, :, 0]
        lrfs = tf.stack([normals, lrfs[:, :, 1], lrfs[:, :, 2]], axis=-1)

    return lrfs, neighborhoods, neighborhood_indices
