import tensorflow as tf


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
    norm = tf.einsum("ij,ij->i", vertices, vertices)
    norm = tf.reshape(norm, (-1, 1)) - 2 * tf.einsum("ik,jk->ij", vertices, vertices) + tf.reshape(norm, (1, -1))

    where_nans = tf.where(tf.math.is_nan(tf.sqrt(norm)))
    norm = tf.tensor_scatter_nd_update(
        norm, where_nans, tf.zeros(shape=(tf.shape(where_nans)[0],), dtype=tf.float32)
    )

    return tf.sqrt(norm)


@tf.function(jit_compile=True)
def group_neighborhoods(vertices, radius, neighbor_limit=32, distance_matrix=None):
    """Finds and groups vertex-neighborhoods for a given radius.

    Collect neighbors in a given radius. From all vertices select the closest 'n_neighbors' many.
    If any of the selected exceeds the maximum radius, its coordinates are set to the maximum distance from
    the origin (keeping those neighbors allows for batching).

    Parameters
    ----------
    vertices: tf.Tensor
        All vertices of a mesh.
    radius: float
        A 1D-tensor containing the radii of each neighborhood. I.e., its first dimension needs to be of the same size
        as the first dimension of the 'vertices'-tensor.
    neighbor_limit: int
        The maximum amount of neighbors per neighborhood.
    distance_matrix: tf.Tensor
        The Euclidean distance matrix for the given vertices.

    Returns
    -------
    tf.Tensor, tf.Tensor:
        A tuple containing two tensors. The first tensor describes vertex coordinates in each neighborhood. The second
        described the vertex indices in each neighborhood.
    """
    # 1.) For each vertex determine local sets of neighbors
    if distance_matrix is None:
        distance_matrix = compute_distance_matrix(vertices)

    # 2.) Get neighborhood vertex indices
    # 'neighborhoods_indices': (vertices, neighbor_limit)
    neighbor_distances, neighborhoods_indices = tf.math.top_k(-distance_matrix, neighbor_limit)
    neighbor_distances = -neighbor_distances

    # 3.) Shift corresponding vertex-coordinates s.t. neighborhood-origin lies in [0, 0, 0].
    # 'vertex_neighborhoods': (vertices, neighbor_limit, 3)
    vertex_neighborhoods = tf.gather(vertices, neighborhoods_indices, axis=0) - tf.expand_dims(vertices, axis=1)

    # 4.) To allow batching: In step (2) we potentially select vertices that are further out than 'radius'.
    #     We set those to the edge of the neighborhood s.t. their weights for LRF computation will be zero.
    set_zero_at = tf.where(neighbor_distances > tf.reshape(radius, (-1, 1)))
    updates = tf.tile(tf.reshape(tf.sqrt((radius ** 2) / 3), (1, 1)), multiples=[tf.shape(set_zero_at)[0], 3])
    vertex_neighborhoods = tf.tensor_scatter_nd_update(vertex_neighborhoods, set_zero_at, updates)

    return vertex_neighborhoods, neighborhoods_indices


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
        A 1D-tensor containing the radii of each neighborhood. I.e., its first dimension needs to be of the same size
        as the first dimension of the 'neighborhoods'-tensor.

    Returns
    -------
    tf.Tensor:
        Local reference frames for all given neighborhoods.
    """
    ###########################
    # 1.) Compute Eigenvectors
    ###########################
    # Calculate neighbor weights
    # 'distance_weights': (vertices, n_neighbors)
    distance_weights = tf.expand_dims(radius, axis=-1) - tf.linalg.norm(neighborhoods, axis=-1)

    # Compute weighted covariance matrices
    # 'weighted_cov': (vertices, 3, 3)
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


@tf.function(jit_compile=True)
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

    # Preserve Euclidean metric between original vertices (geodesic distance approximation)
    return tf.expand_dims(tf.linalg.norm(neighborhoods, axis=-1), axis=-1) * tf.math.l2_normalize(projections, axis=-1)
