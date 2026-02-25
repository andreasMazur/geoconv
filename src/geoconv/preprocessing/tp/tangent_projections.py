from geoconv.utils.misc import compute_distance_matrix

import numpy as np


def disambiguate_axis(axis, neighborhood):
    """Disambiguate axes returned by local Eigenvalue analysis.

    Disambiguation follows the formal procedure as described in:
    > [SHOT: Unique signatures of histograms for surface and texture
     description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    axis: np.array
        The axis to be disambiguated.
    neighborhood: np.array
        The neighborhood the axis represents.

    Returns
    -------
    np.array:
        The according to the neighborhood disambiguated axis.
    """
    s_plus = np.sum(np.einsum("nc,c->n", neighborhood, axis) >= 0)
    s_minus = np.sum(np.einsum("nc,c->n", neighborhood, -axis) > 0)
    return axis if s_plus >= s_minus else -axis


def get_2d_projections(neighborhood_3d, rotation_axis=None, x_axis=None, y_axis=None, z_axis=None, rescale=True):
    """Projects a 3D neighborhood into a 2D tangent plane via local covariance analysis.

    Parameters
    ----------
    neighborhood_3d: np.ndarray
        The 3D neighborhood to be projected.
    rotation_axis: np.ndarray
        The direction into which the z-axis should point.
    x_axis: np.ndarray
        The x-axis of the local reference frame that shall be used. If not given, entire LRF will be computed.
    y_axis: np.ndarray
        The y-axis of the local reference frame that shall be used. If not given, entire LRF will be computed.
    z_axis: np.ndarray
        The z-axis of the local reference frame that shall be used. If not given, entire LRF will be computed.
    rescale: bool
        Rescale projections to their original Euclidean length.

    Return
    ------
    np.ndarray:
        The 2D projections of the neighborhood.
    """
    # If neighborhood contains only one element return the origin
    if neighborhood_3d.shape[0] == 1:
        return np.array([[0., 0.]])

    # Determine local reference frame via covariance analysis if not given
    if x_axis is None or y_axis is None or z_axis is None:
        # Determine weights for covariance matrix
        neighbor_distances = np.linalg.norm(neighborhood_3d, axis=-1)
        chart_neighborhood_radius = np.linalg.norm(neighborhood_3d, axis=-1).max()
        neighbor_weights = chart_neighborhood_radius - neighbor_distances

        # Compute distance-weighted covariance matrix
        cov_matrix = 1 / neighbor_weights.sum() * np.einsum(
            "n,ni,nj->ij", neighbor_weights, neighborhood_3d, neighborhood_3d
        )

        # Get eigenvalues and -vectors and disambiguate axes according to SHOT-paper
        # Smallest eigenvalue: eigenvalues[0]
        # Associated eigenvector: eigenvectors[:, 0]
        _, eigenvectors = np.linalg.eigh(cov_matrix)
        z_axis = disambiguate_axis(eigenvectors[:, 0], neighborhood_3d)  # (associated to the smallest eigenvalue)
        if rotation_axis is not None:
            dot_product = np.einsum("i,i->", z_axis, rotation_axis)
            if dot_product < 0:
                z_axis = -z_axis
        x_axis = disambiguate_axis(eigenvectors[:, 2], neighborhood_3d)  # (associated to the largest eigenvalue)
        y_axis = np.cross(z_axis, x_axis)

    # Project neighborhood into tangent plane (plane spanned by x- and y-axis)
    projections = neighborhood_3d - np.einsum("c,nc->n", z_axis, neighborhood_3d)[:, None] * z_axis[None, :]

    # Basis change (first dim is zero, as we have projected into plane of z-axis)
    projections = np.einsum(
        "ij,nj->ni", np.linalg.inv(np.array([z_axis, y_axis, x_axis]).T), projections
    )[:, 1:]

    # Rescale projections to Euclidean length
    if rescale:
        projections = np.concatenate(
            [np.zeros((1, 2)), projections[1:] / (np.linalg.norm(projections[1:] + np.finfo(np.float32).eps, axis=-1, keepdims=True))], axis=0
        )
        projections = projections * np.linalg.norm(neighborhood_3d, axis=-1, keepdims=True)
    return projections


def get_3d_neighborhood(vertices, max_radius, return_as_array=False, required_origins=None):
    """Determines the 3D neighborhood of the closest 'max_neighbors' vertices for all given 3D vertices.

    Parameters
    ----------
    vertices: np.ndarray
        The vertices of the shape.
    max_radius: float
        The maximum number of neighbors per neighborhood.
    return_as_array: bool
        Whether to return the neighborhoods in a padded array.
    required_origins: np.ndarray
        A list of indices for neighborhood-origins to keep.

    Returns
    -------
    np.ndarray:
        An array of shape (vertices, max_neighbors, 3) that contains all neighborhoods.
    """
    # 1.) Compute Euclidean distances among shape vertices
    # 'distance_matrix': (vertices, vertices)
    distance_matrix = compute_distance_matrix(vertices)

    # 2.) Define a (vertices, vertices, 3) 3D coordinates array
    neighborhoods = np.tile(vertices[None], (vertices.shape[0], 1, 1))

    # 3.) Shift neighborhoods into (0, 0, 0)
    neighborhoods = neighborhoods - neighborhoods[np.arange(vertices.shape[0]), np.arange(vertices.shape[0])][:, None]

    if return_as_array:
        # 4.) Set all 3D coordinates farther than 'max_radius' to 'np.inf'
        neighbor_indices = np.where(distance_matrix > max_radius)

        # 5.) Return neighborhoods as array
        neighborhoods[neighbor_indices] = [np.inf, np.inf, np.inf]
        if required_origins is not None:
            return np.array(neighborhoods)[required_origins]
        else:
            return np.array(neighborhoods)
    else:
        # 4.) Select all 3D coordinates closer than 'max_radius'
        neighbor_indices = np.where(distance_matrix <= max_radius)

        # 5.) Return neighborhoods as list
        hood_list, current_hood_idx = [], -1
        for hood_idx, neigh_id in np.stack(neighbor_indices, axis=-1):
            # 5.1) Case: only a subset of neighborhoods is wished for
            if required_origins is not None and hood_idx in required_origins:
                if hood_idx != current_hood_idx:
                    hood_list.append([])
                    current_hood_idx += 1
                hood_list[-1].append(np.array(neighborhoods[hood_idx, neigh_id]))

            # 5.1) Case: All neighborhoods are wished for
            elif required_origins is None:
                if hood_idx != current_hood_idx:
                    hood_list.append([])
                    current_hood_idx += 1
                hood_list[-1].append(np.array(neighborhoods[hood_idx, neigh_id]))

        # 6.) Filter neighborhood indices down to required indices
        if required_origins is not None:
            neighbor_indices = np.array(
                [(h, n) for (h, n) in np.stack(neighbor_indices, axis=-1) if h in required_origins]
            )
            neighbor_indices = (neighbor_indices[:, 0] - neighbor_indices[0, 0], neighbor_indices[:, 1])
        return hood_list, neighbor_indices
