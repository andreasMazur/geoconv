import torch


def tensor_scatter_nd_update_(dest, indices, updates):
    """In-place port of tf.tensor_scatter_nd_update.

    Parameters
    ----------
    dest: torch.Tensor
        The tensor to be updated.
    indices: torch.tensor
        The indices of the tensor where updates should be made.
    updates: torch.Tensor
        The updates to insert at the indices.
    """
    dest[indices.unbind(dim=-1)] = updates


def tensor_scatter_nd_add_(dest, indices, updates):
    """In-place port of tf.tensor_scatter_nd_update.

    Parameters
    ----------
    dest: torch.Tensor
        The tensor to be updated.
    indices: torch.tensor
        The indices of the tensor where updates should be made.
    updates: torch.Tensor
        The updates to insert at the indices.
    """
    # Add accumulated updates
    dest.index_put_(tuple(indices.unbind(dim=-1)), updates, accumulate=True)


def compute_distance_matrix(vertices):
    """Computes the pair-wise Euclidean distances between all pairs of vertices.

    Parameters
    ----------
    vertices: torch.Tensor
        A tensor of shape (batch, vertices, 3), containing all 3D vertices for a batch of shapes. torch.Tensor

    Returns
    -------
    torch.Tensor:
        A tensor of shape (batch, vertices, vertices) containing the pair-wise Euclidean distances between all
        pairs of vertices.
    """
    # 'norm': (batch, vertices)
    vertices = vertices.to(torch.float64)
    norm = torch.einsum("bij,bij->bi", vertices, vertices)

    # 'norm': (batch, vertices, vertices)
    batch_size = vertices.size(0)
    norm = (
        norm.reshape(batch_size, -1, 1)
        - 2 * torch.einsum("bik,bjk->bij", vertices, vertices)
        + norm.reshape(batch_size, 1, -1)
    )

    # 'where_nans': (n_nans, 3)
    where_nans = torch.nonzero(torch.isnan(torch.sqrt(norm)), as_tuple=False)
    tensor_scatter_nd_update_(
        norm,
        where_nans,
        torch.zeros(where_nans.size(0), dtype=torch.float64, device=norm.device)
    )

    # 'return': (batch, vertices, vertices)
    return torch.sqrt(norm).to(torch.float32)


def disambiguate_axes(neighborhoods, eigen_vectors):
    """Disambiguate LRF directions according to the rule-set given in the original paper.

    Original paper:
    > SHOT: Unique signatures of histograms for surface and texture description.
    > Samuele Salti, Federico Tombari, and Luigi Di Stefano.
    > DOI: 10.1016/j.cviu.2014.04.011

    Parameters
    ----------
    neighborhoods: torch.Tensor
        A tensor of shape (batch, vertices, k_neighbors, 3) containing Euclidean neighborhoods centered in (0, 0, 0).
    eigen_vectors: torch.Tensor
        A tensor of shape (batch, vertices, 3) containing the eigen vectors whose direction shall be disambiguated.

    Returns
    -------
    torch.Tensor:
        A tensor of shape (batch, vertices, 3) containing the directionally disambiguated eigen vectors.
    """
    # 'neg_eigen_vectors': (batch, vertices, 3)
    neg_eigen_vectors = -eigen_vectors

    # 'ev_count': (batch, vertices)
    ev_count = torch.count_nonzero(
        torch.einsum("bnvk,bnk->bnv", neighborhoods, eigen_vectors) >= 0.0, dim=-1
    )

    # 'ev_neg_count': (batch, vertices)
    ev_neg_count = torch.count_nonzero(
        torch.einsum("bnvk,bnk->bnv", neighborhoods, neg_eigen_vectors) > 0.0, dim=-1
    )

    # 'return': (batch, vertices, 3)
    return torch.where((ev_count >= ev_neg_count)[..., None], eigen_vectors, neg_eigen_vectors)


def shot_lrf(neighborhoods, radii):
    """Computes SHOT LRF for given Euclidean neighborhoods (epsilon-balls) and their maximum radii (epsilon).

    Parameters
    ----------
    neighborhoods: torch.tensor
        A tensor of shape (batch, vertices, k_neighbors, 3) containing Euclidean neighborhoods centered in (0, 0, 0).
    radii: torch.tensor
        A tensor of shape (batch, vertices) containing the neighborhood radii associated to the 'neighborhoods' tensor.

    Returns
    -------
    torch.tensor:
        A tensor of shape (batch, vertices, 3, 3) containing the SHOT LRF per neighborhood.
    """
    # 1.) Compute Eigenvectors
    # Calculate neighbor weights
    # 'distance_weights': (batch, vertices, n_neighbors)
    distance_weights = radii.unsqueeze(dim=-1) - torch.linalg.norm(neighborhoods, dim=-1)

    # Compute weighted covariance matrices for each neighborhood
    # 'weighted_cov': (batch, vertices, 3, 3)
    weighted_cov = torch.einsum("bnv,bnvi,bnvj->bnij", distance_weights, neighborhoods, neighborhoods)

    # 2.) Disambiguate axes
    # First eigen vector corresponds to smallest eigen value (i.e. plane normal)
    # 'eigen_vectors': (batch, vertices, 3, 3)
    _, eigen_vectors = torch.linalg.eigh(weighted_cov)

    # Columns contain eigenvectors
    # '?_axes': (batch, vertices, 3)
    x_axes = disambiguate_axes(neighborhoods, eigen_vectors[..., 2])
    z_axes = disambiguate_axes(neighborhoods, eigen_vectors[..., 0])
    y_axes = torch.linalg.cross(z_axes, x_axes)

    # 'return': (batch, vertices, 3, 3)
    return torch.stack([z_axes, y_axes, x_axes], dim=-1)


def logarithmic_map(lrfs, neighborhoods):
    """Computes projections of neighborhoods into their local reference frames.

    Parameters
    ----------
    lrfs: torch.Tensor
        A tensor of shape (vertices, 3, 3) containing the axes of local reference frames.
    neighborhoods: torch.Tensor
        A tensor of shape (vertices, n_neighbors, 3) containing the neighborhoods around all vertices.

    Returns
    -------
    torch.Tensor:
        A tensor of shape (vertices, n_neighbors, 2) containing the coordinates of the neighbor-projections
        within the tangent plane. Euclidean distance are preserved and used as an approximate to geodesic distances.
    """
    # Get tangent plane normals (z-axes of lrfs)
    # 'normals': (batch, vertices, 3)
    normals = lrfs[..., 0]

    # Compute tangent plane projections (logarithmic map)
    # 'scaled_normals': (batch, vertices, n_neighbors, 3)
    scaled_normals = neighborhoods @ torch.unsqueeze(normals, dim=-1) * torch.unsqueeze(normals, dim=2)

    # 'projections': (batch, vertices, n_neighbors, 3)
    projections = neighborhoods - scaled_normals

    # Basis change of neighborhoods into lrf coordinates
    # Plane projection cause first dimension to be 0 => Remove it
    # 'projections': (batch, vertices, n_neighbors, 2)
    projections = torch.einsum("bvij,bvnj->bvni", torch.linalg.inv(lrfs), projections)[..., 1:]

    # Use 'projection / adjacent * hypotenuse' as estimate to geodesic distance
    adj = torch.linalg.norm(projections, dim=-1)
    hy = torch.linalg.norm(neighborhoods, dim=-1)

    zero_indices = torch.nonzero(adj == 0.0, as_tuple=False)
    tensor_scatter_nd_update_(adj, zero_indices, torch.ones(zero_indices.size(0), device=adj.device))
    tensor_scatter_nd_update_(hy, zero_indices, torch.ones(zero_indices.size(0), device=hy.device))

    # Rescale projections to their original Euclidean distances
    return projections / adj[..., None] * hy[..., None]


def compute_neighborhood(k_neighbors, vertices):
    """Computes Euclidean neighborhoods (epsilon-balls) around all given vertices.

    Parameters
    ----------
    k_neighbors: int
        The number of neighbors to put into one neighborhood.
    vertices: torch.Tensor
        A tensor of shape (batch, vertices, 3), containing all 3D vertices for a batch of shapes. torch.Tensor

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor):
        Three tensors of shapes (batch, vertices, k_neighbors, 3), (batch, vertices, k_neighbors) and (batch, vertices)
        containing the Euclidean neighborhoods centered in (0, 0, 0), the global shape indices of the vertices
        contained in the neighborhoods and the neighborhood radii, i.e., their maximum Euclidean distance to a neighbor.
    """
    # 1.) Compute radius for local parameterization spaces.
    # 'distance_matrix': (batch, vertices, vertices)
    distance_matrix = compute_distance_matrix(vertices)

    # 'sorted_indices': (batch, vertices)
    sorted_indices = torch.argsort(distance_matrix, dim=-1)[..., k_neighbors]

    # 'batch_idx': (batch, 1)
    batch_indices = torch.arange(vertices.size(0), device=vertices.device)[:, None]

    # 'vertex_idx': (1, vertices)
    vertex_indices = torch.arange(vertices.size(1), device=vertices.device)[None, :]

    # 'radii': (batch, vertices)
    radii = distance_matrix[batch_indices, vertex_indices, sorted_indices]

    # 2.) Get vertex-neighborhoods
    # 'neighborhood_indices': (batch, vertices, k_neighbors)
    _, neighborhood_indices = torch.topk(-distance_matrix, k_neighbors, dim=-1)

    # 'neighborhoods': (batch, vertices, k_neighbors, 3)
    neighborhoods = vertices[batch_indices[..., None], neighborhood_indices] - vertices.unsqueeze(-2)

    # 'neighborhoods':        (batch, vertices, k_neighbors, 3)
    # 'neighborhood_indices': (batch, vertices, k_neighbors)
    # 'radii':                (batch, vertices)
    return neighborhoods, neighborhood_indices, radii


def knn_shot_lrf(k_neighbors, vertices):
    """Computes SHOT local reference frames (LRF).

    Original paper:
    > SHOT: Unique signatures of histograms for surface and texture description.
    > Samuele Salti, Federico Tombari, and Luigi Di Stefano.
    > DOI: 10.1016/j.cviu.2014.04.011

    Parameters
    ----------
    k_neighbors: int
        The number of neighbors to consider when creating Euclidean neighborhoods for LRF construction.
    vertices: torch.Tensor
        A tensor of shape (batch, vertices, 3), containing all 3D vertices for a batch of shapes.

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor):
        Three tensors of shapes (batch, vertices, 3, 3), (batch, vertices, k_neighbors, 3) and
        (batch, vertices, k_neighbors) containing the LRFs, Euclidean neighborhoods centered in (0, 0, 0) and the
        global shape indices of the vertices contained in the neighborhoods.
    """
    # 1.) Compute neighborhoods
    # 'neighborhoods':        (batch, vertices, k_neighbors, 3)
    # 'neighborhood_indices': (batch, vertices, k_neighbors)
    # 'radii':                (batch, vertices)
    neighborhoods, neighborhood_indices, radii = compute_neighborhood(k_neighbors=k_neighbors, vertices=vertices)

    # 2.) Get local reference frames
    # 'lrfs': (batch, vertices, 3, 3)
    lrfs = shot_lrf(neighborhoods, radii)

    # 3.) Return results
    # 'lrfs';                 (batch, vertices, 3, 3)
    # 'neighborhoods':        (batch, vertices, k_neighbors, 3)
    # 'neighborhood_indices': (batch, vertices, k_neighbors)
    return lrfs, neighborhoods, neighborhood_indices
