from typing import Tuple

import torch


@torch.jit.script
def compute_distance_matrix(vertices: torch.FloatTensor) -> torch.FloatTensor:
    """Computes the Euclidean distance between given vertices.

    Parameters
    ----------
    vertices: torch.Tensor
        The vertices to compute the distance between.

    Returns
    -------
    torch.Tensor:
        A square distance matrix for the given vertices.
    """
    vertices = vertices.to(torch.float64)

    norm = torch.einsum("ij,ij->i", vertices, vertices)
    norm = (
        norm.reshape((-1, 1))
        - 2 * torch.einsum("ik,jk->ij", vertices, vertices)
        + norm.reshape((1, -1))
    )

    norm = torch.nan_to_num(torch.sqrt(norm))

    return norm.to(torch.float32)


@torch.jit.script
def disambiguate_axes(
    neighborhood_vertices: torch.Tensor, eigen_vectors: torch.Tensor
) -> torch.Tensor:
    """Disambiguate axes returned by local Eigenvalue analysis.

    Disambiguation follows the formal procedure as described in:
    > [SHOT: Unique signatures of histograms for surface and texture
     description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    neighborhood_vertices: torch.Tensor
        The vertices of the neighborhoods.
    eigen_vectors: torch.Tensor
        The Eigenvectors of all neighborhoods for one dimension, i.e. it has size (#neighborhoods, 3).
        E.g. the x-axes.

    Returns
    -------
    torch.Tensor:
        The disambiguated Eigenvectors.
    """
    neg_eigen_vectors = -eigen_vectors
    ev_count = (
        torch.einsum("nvk,nk->nv", neighborhood_vertices, eigen_vectors) >= 0
    ).sum(dim=-1)
    ev_neg_count = (
        torch.einsum("nvk,nk->nv", neighborhood_vertices, -eigen_vectors) > 0.0
    ).sum(dim=-1)
    mask = (ev_count >= ev_neg_count).unsqueeze(-1)
    return torch.where(mask, eigen_vectors, neg_eigen_vectors)


@torch.jit.script
def shot_lrf(neighborhoods: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    """Computes SHOT local reference frames.

    SHOT computation was introduced in:
    > [SHOT: Unique signatures of histograms for surface and texture
     description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    neighborhoods: torch.Tensor
        The vertices of the neighborhoods shifted around the neighborhood origin.
    radii: torch.Tensor
        A 1D-tensor containing the radii of each neighborhood. I.e., its first dimension needs to be of the same size
        as the first dimension of the 'neighborhoods'-tensor.

    Returns
    -------
    torch.Tensor:
        Local reference frames for all given neighborhoods.
    """
    # 1.) Compute Eigenvectors
    # Calculate neighbor weights
    # 'distance_weights': (vertices, n_neighbors)
    distance_weights = torch.unsqueeze(radii, dim=-1) - torch.linalg.norm(
        neighborhoods, dim=-1
    )

    # Compute weighted covariance matrices
    # 'weighted_cov': (vertices, 3, 3)
    weighted_cov = torch.einsum(
        "nv,nvi,nvj->nij", distance_weights, neighborhoods, neighborhoods
    )

    # 2.) Disambiguate axes
    # First eigen vector corresponds to smallest eigen value (i.e. plane normal)
    _, eigen_vectors = torch.linalg.eigh(weighted_cov)

    # Columns contain eigenvectors
    x_axes = disambiguate_axes(neighborhoods, eigen_vectors[:, :, 2])
    z_axes = disambiguate_axes(neighborhoods, eigen_vectors[:, :, 0])
    y_axes = torch.linalg.cross(z_axes, x_axes)

    return torch.stack([z_axes, y_axes, x_axes], dim=-1)


@torch.jit.script
def logarithmic_map(lrfs: torch.Tensor, neighborhoods: torch.Tensor) -> torch.Tensor:
    """Computes projections of neighborhoods into their local reference frames.

    Parameters
    ----------
    lrfs: torch.Tensor
        A 3D-tensor of shape (vertices, 3, 3) that contains the axes of local reference frames.
    neighborhoods: torch.Tensor
        A 3D-tensor of shape (vertices, n_neighbors, 3) that contains the neighborhoods around all vertices.

    Returns
    -------
    torch.Tensor:
        A 3D-tensor of shape (vertices, n_neighbors, 2) that contains the coordinates of the neighbor-projections
        within the tangent plane. Euclidean distance are preserved and used as an approximate to geodesic distances.
    """
    # Get tangent plane normals (z-axes of lrfs)
    normals = lrfs[:, 0, :]

    # Compute tangent plane projections (logarithmic map)
    scaled_normals = (
        neighborhoods @ torch.unsqueeze(normals, dim=-1) * torch.unsqueeze(normals, dim=1)
    )
    projections = neighborhoods - scaled_normals

    # Basis change of neighborhoods into lrf coordinates
    projections = torch.einsum(
        "vij,vnj->vni", torch.linalg.inv(lrfs.permute([0, 2, 1])), projections
    )[:, :, 1:]

    # Use 'projection / adjacent * hypotenuse' as estimate to geodesic distance
    adj, hy = torch.linalg.norm(projections, dim=-1), torch.linalg.norm(
        neighborhoods, dim=-1
    )
    adj = torch.where(adj == 0.0, 1.0, adj)
    hy = torch.where(adj == 0, 1.0, hy)

    # Rescale projections to their original Euclidean distances
    projections = projections / adj[..., None] * hy[..., None]

    return projections


@torch.jit.script
def knn_shot_lrf(
    k_neighbors: int, vertices: torch.Tensor, repetitions: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1.) Compute radius for local parameterization spaces. Keep it equal for all for comparability.
    # 'distance_matrix': (vertices, vertices)
    # 'radii': (vertices,)
    distance_matrix = compute_distance_matrix(vertices)
    radii = distance_matrix.gather(
        1, distance_matrix.argsort(dim=-1)[:, k_neighbors].unsqueeze(1)
    ).squeeze(1)

    # 2.) Get vertex-neighborhoods
    # 'neighborhoods': (vertices, n_neighbors, 3)
    values, neighborhood_indices = torch.topk(-distance_matrix, k_neighbors, dim=-1)
    neighborhoods = vertices[neighborhood_indices] - vertices.unsqueeze(1)

    # 3.) Get local reference frames
    # 'lrfs': (vertices, 3, 3)
    lrfs = shot_lrf(neighborhoods, radii)

    # 4.) Make normal vectors point away from centroid (outwards from shape)
    signs = -(
        (
            torch.einsum("vi,vi->v", lrfs[:, :, 0], vertices.mean(dim=0) - vertices)
            >= 0
        ).int()
    )
    signs = signs + (signs == 0).int()
    normals = signs.float().unsqueeze(-1) * lrfs[:, :, 0]
    lrfs = torch.stack([normals, lrfs[:, :, 1], lrfs[:, :, 2]], dim=-1)

    # 5.) Make normal vectors in neighborhoods point the same direction
    # (non-convex shapes -> "outwards" might differ locally)
    for rep in range(repetitions):
        normals = lrfs[:, :, 0][neighborhood_indices[:, 1:]]
        signs = -(
            (
                torch.sum(
                    (torch.einsum("vi,vni->vn", lrfs[:, :, 0], normals) >= 0).int(),
                    dim=-1,
                )
                <= k_neighbors // 2
            ).int()
        )
        signs = signs + (signs == 0).int()
        normals = signs.float().unsqueeze(-1) * lrfs[:, :, 0]
        lrfs = torch.stack([normals, lrfs[:, :, 1], lrfs[:, :, 2]], dim=-1)

    return lrfs, neighborhoods, neighborhood_indices
