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
    vertices = tf.cast(vertices, tf.float64)

    norm = tf.einsum("bij,bij->bi", vertices, vertices)

    batch_size = tf.shape(vertices)[0]
    norm = (
        tf.reshape(norm, (batch_size, -1, 1))
        - 2 * tf.einsum("bik,bjk->bij", vertices, vertices)
        + tf.reshape(norm, (batch_size, 1, -1))
    )

    where_nans = tf.where(tf.math.is_nan(tf.sqrt(norm)))
    norm = tf.tensor_scatter_nd_update(
        norm, where_nans, tf.zeros(shape=(tf.shape(where_nans)[0],), dtype=tf.float64)
    )

    return tf.cast(tf.sqrt(norm), tf.float32)


@tf.function(jit_compile=True)
def disambiguate_axes(neighborhoods, eigen_vectors):
    """Disambiguate axes returned by local Eigenvalue analysis.

    Disambiguation follows the formal procedure as described in:
    > [SHOT: Unique signatures of histograms for surface and texture
     description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    neighborhoods: tf.Tensor
        The vertices of the neighborhoods.
    eigen_vectors: tf.Tensor
        The Eigenvectors of all neighborhoods for one dimension, i.e. it has size (batch, neighborhoods, 3).
        E.g. the x-axes.

    Returns
    -------
    tf.Tensor:
        The disambiguated Eigenvectors.
    """
    neg_eigen_vectors = -eigen_vectors
    ev_count = tf.math.count_nonzero(
        tf.einsum("bnvk,bnk->bnv", neighborhoods, eigen_vectors) >= 0.0, axis=-1
    )
    ev_neg_count = tf.math.count_nonzero(
        tf.einsum("bnvk,bnk->bnv", neighborhoods, -eigen_vectors) > 0.0, axis=-1
    )

    # return (batch, vertices, 3)
    return tf.gather(
        tf.stack([neg_eigen_vectors, eigen_vectors], axis=2),
        tf.cast(ev_count >= ev_neg_count, tf.int32),
        batch_dims=2,
    )


@tf.function(jit_compile=True)
def shot_lrf(neighborhoods, radii):
    """Computes SHOT local reference frames.

    SHOT computation was introduced in:
    > [SHOT: Unique signatures of histograms for surface and texture
     description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    The z-axis (point-normals) can be by:
        'returned_tensor[..., 0]'
    ... the y-axis by:
        'returned_tensor[..., 1]'
    ... the x-axis by:
        'returned_tensor[..., 2]'

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
    # 'distance_weights': (batch, vertices, n_neighbors)
    distance_weights = tf.expand_dims(radii, axis=-1) - tf.linalg.norm(neighborhoods, axis=-1)

    # Compute weighted covariance matrices
    # 'weighted_cov': (batch, vertices, 3, 3)
    weighted_cov = tf.einsum("bnv,bnvi,bnvj->bnij", distance_weights, neighborhoods, neighborhoods)

    # 2.) Disambiguate axes
    # First eigen vector corresponds to smallest eigen value (i.e. plane normal)
    # 'eigen_vectors': (batch, vertices, 3, 3)
    _, eigen_vectors = tf.linalg.eigh(weighted_cov)

    # Columns contain eigenvectors
    x_axes = disambiguate_axes(neighborhoods, eigen_vectors[..., 2])
    z_axes = disambiguate_axes(neighborhoods, eigen_vectors[..., 0])
    y_axes = tf.linalg.cross(z_axes, x_axes)

    return tf.stack([z_axes, y_axes, x_axes], axis=-1)


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
    # 'normals': (batch, vertices, 3)
    normals = lrfs[..., 0]

    # Compute tangent plane projections (logarithmic map)
    # 'scaled_normals': (batch, vertices, n_neighbors, 3)
    scaled_normals = (
        neighborhoods @ tf.expand_dims(normals, axis=-1) * tf.expand_dims(normals, axis=2)
    )

    # 'projections': (batch, vertices, n_neighbors, 3)
    projections = neighborhoods - scaled_normals

    # Basis change of neighborhoods into lrf coordinates
    # Plane projection cause first dimension to be 0 => Remove it
    # 'projections': (batch, vertices, n_neighbors, 2)
    projections = tf.einsum("bvij,bvnj->bvni", tf.linalg.inv(lrfs), projections)[..., 1:]

    # Use 'projection / adjacent * hypotenuse' as estimate to geodesic distance
    adj = tf.linalg.norm(projections, axis=-1)
    hy = tf.linalg.norm(neighborhoods, axis=-1)

    zero_indices = tf.cast(tf.where(adj == 0.0), tf.int32)
    adj = tf.tensor_scatter_nd_update(
        adj, zero_indices, tf.ones((tf.shape(zero_indices)[0],))
    )
    hy = tf.tensor_scatter_nd_update(
        hy, zero_indices, tf.ones((tf.shape(zero_indices)[0],))
    )

    # Rescale projections to their original Euclidean distances
    return projections / adj[..., None] * hy[..., None]


@tf.function(jit_compile=True)
def compute_neighborhood(vertices, k_neighbors):
    """Determines the Euclidean k-neighborhood around all given vertices.

    Parameters
    ----------
    vertices: tf:tensor
        A tensor containing all mesh vertices
    k_neighbors: int
        The number of neighbors to consider around each vertex.

    Returns
    -------
    (tf.Tensor, tf.Tensor, tf.Tensor):
        A tensor of shape [batch, vertices, n_neighbors, 3], containing all k neighbors around each vertex. Another
        tensor of shape [batch, vertices, n_neighbors], containing the original indices of the vertices in the
        neighborhoods. A tensor of shape [batch, vertices] containing the Euclidean radii of each neighborhood.
    """
    # 1.) Compute radius for local parameterization spaces.
    # 'distance_matrix': (batch, vertices, vertices)
    distance_matrix = compute_distance_matrix(vertices)

    # 'radii': (batch, vertices)
    radii = tf.gather(
        distance_matrix,
        tf.argsort(distance_matrix, axis=-1)[..., k_neighbors],
        batch_dims=2,
    )

    # 2.) Get vertex-neighborhoods
    # 'neighborhoods': (batch, vertices, n_neighbors, 3)
    _, neighborhood_indices = tf.math.top_k(-distance_matrix, k_neighbors)
    neighborhoods = tf.gather(vertices, neighborhood_indices, batch_dims=1) - vertices[..., None, :]

    return neighborhoods, neighborhood_indices, radii


@tf.function(jit_compile=True)
def knn_shot_lrf(k_neighbors, vertices):
    """Computes the local reference frames of SHOT-descriptors.

    Original paper:
    > SHOT: Unique signatures of histograms for surface and texture description
    > Samuele Salti and Federico Tombari and Luigi Di Stefano
    > DOI: 10.1016/j.cviu.2014.04.011

    Parameters
    ----------
    k_neighbors: int
        The amount of neighbors to consider around each vertex.
    vertices: tf.Tensor
        The mesh vertices.

    Returns
    -------
    (tf.Tensor, tf.Tensor, tf.Tensor):
        A tensor of shape [batch, n_vertices, 3, 3], containing all LRFs around each vertex. Another
        tensor of shape [batch, vertices, n_neighbors, 3], containing all k neighbors around each vertex.
        A tensor of shape [batch, vertices] containing the Euclidean radii of each neighborhood. A last
        tensor of shape [batch, vertices, n_neighbors], containing the original indices of the vertices in the
        neighborhoods.
    """
    # 1.) Compute neighborhoods
    neighborhoods, neighborhood_indices, radii = compute_neighborhood(
        vertices, k_neighbors
    )

    # 2.) Get local reference frames
    # 'lrfs': (batch, vertices, 3, 3)
    lrfs = shot_lrf(neighborhoods, radii)

    return lrfs, neighborhoods, neighborhood_indices
