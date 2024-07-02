import tensorflow as tf


@tf.function
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
    norm = tf.einsum("ij,ij->i", vertices, vertices)
    norm = tf.reshape(norm, (-1, 1)) - 2 * tf.einsum("ik,jk->ij", vertices, vertices) + tf.reshape(norm, (1, -1))

    where_nans = tf.where(tf.math.is_nan(tf.sqrt(norm)))
    norm = tf.tensor_scatter_nd_update(
        norm, where_nans, tf.zeros(shape=(tf.shape(where_nans)[0],), dtype=tf.float32)
    )

    return tf.sqrt(norm)


@tf.function
def group_neighborhoods(vertices, radius, distance_matrix=None):
    """Finds and groups vertex-neighborhoods for a given radius.

    Parameters
    ----------
    vertices: tf.Tensor
        All vertices of a mesh.
    radius: float
        The radius of a neighborhood.
    distance_matrix: tf.Tensor
        The Euclidean distance matrix for the given vertices.

    Returns
    -------
    tf.Tensor:

    """
    # 1.) For each vertex determine local sets of neighbors
    if distance_matrix is None:
        distance_matrix = compute_distance_matrix(vertices)
    neighborhood_mask = distance_matrix <= tf.cast(radius, tf.float32)

    # 2.) Get neighborhood vertex indices (with zero padding accounting for different amount of vertices)
    indices = tf.where(neighborhood_mask)
    neighborhoods_indices = tf.RaggedTensor.from_value_rowids(
        values=indices[:, 1], value_rowids=indices[:, 0]
    ).to_tensor(default_value=-1)

    # 3.) Shift corresponding vertex-coordinates s.t. neighborhood-origin lies in [0, 0, 0].
    vertex_neighborhoods = tf.gather(vertices, neighborhoods_indices, axis=0) - tf.expand_dims(vertices, axis=1)

    # 4.) Set fill coordinates to edge of neighborhood s.t. their weights will be zero
    set_zero_at = tf.where(neighborhoods_indices == -1)
    updates = tf.cast(tf.fill(dims=(tf.shape(set_zero_at)[0], 3), value=tf.sqrt((radius ** 2) / 3)), tf.float32)
    vertex_neighborhoods = tf.tensor_scatter_nd_update(vertex_neighborhoods, set_zero_at, updates)

    return vertex_neighborhoods, neighborhoods_indices


@tf.function
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
        The disambiguated Eigenvalues.
    """
    neg_eigen_vectors = -eigen_vectors
    ev_count = tf.math.count_nonzero(tf.einsum("nvk,nk->nv", neighborhood_vertices, eigen_vectors) >= 0, axis=-1)
    ev_neg_count = tf.math.count_nonzero(tf.einsum("nvk,nk->nv", neighborhood_vertices, -eigen_vectors) > 0., axis=-1)
    return tf.gather(
        tf.stack([neg_eigen_vectors, eigen_vectors], axis=1), tf.cast(ev_count >= ev_neg_count, tf.int32), batch_dims=1
    )


@tf.function
def shot_lrf(neighborhoods, radius):
    """Computes SHOT local reference frames.

    SHOT computation was introduced in:
    > [SHOT: Unique signatures of histograms for surface and texture
     description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    neighborhoods: tf.Tensor
        The vertices of the neighborhoods shifted around the neighborhood origin.
    radius: float
        The radius of the neighborhoods.

    Returns
    -------
    tf.Tensor:
        Local reference frames for all given neighborhoods.
    """
    ###########################
    # 1.) Compute Eigenvectors
    ###########################
    # Calculate neighbor weights
    distance_weights = tf.cast(radius, tf.float32) - tf.linalg.norm(neighborhoods, axis=-1)

    # Compute weighted covariance matrices
    weighted_cov = tf.reshape(1 / tf.reduce_sum(distance_weights, axis=-1), (-1, 1, 1)) * tf.einsum(
        "nv,nvi,nvj->nij", distance_weights, neighborhoods, neighborhoods
    )

    ########################
    # 2.) Disambiguate axes
    ########################
    # First eigen vector corresponds to smallest eigen value (i.e. plane normal)
    _, eigen_vectors = tf.linalg.eigh(weighted_cov)
    x_axes = disambiguate_axes(neighborhoods, eigen_vectors[:, 2, :])
    z_axes = disambiguate_axes(neighborhoods, eigen_vectors[:, 0, :])
    y_axes = tf.linalg.cross(z_axes, x_axes)

    return tf.stack([z_axes, y_axes, x_axes], axis=1)
