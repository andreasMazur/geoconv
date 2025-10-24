from tqdm import tqdm

import numpy as np


def get_2d_projections(neighborhood_3d, rotation_axis=None, x_axis=None, y_axis=None, z_axis=None):
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
        z_axis = disambiguate_axis(eigenvectors[:, 0], neighborhood_3d)  # (associated to smallest eigenvalue)
        if rotation_axis is not None:
            dot_product = np.einsum("i,i->", z_axis, rotation_axis)
            if dot_product < 0:
                z_axis = -z_axis
        x_axis = disambiguate_axis(eigenvectors[:, 2], neighborhood_3d)  # (associated to largest eigenvalue)
        y_axis = np.cross(z_axis, x_axis)

    # Project neighborhood into tangent plane (plane spanned by x- and y-axis)
    projections = neighborhood_3d - np.einsum("c,nc->n", z_axis, neighborhood_3d)[:, None] * z_axis[None, :]

    # Basis change (first dim is zero, as we have projected into plane of z-axis)
    projections = np.einsum(
        "ij,nj->ni", np.linalg.inv(np.array([z_axis, y_axis, x_axis]).T), projections
    )[:, 1:]

    return projections


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


def compute_angles(triangle_mesh, distance_charts):
    """Computes the angles via tangent plane projections.

    Computation of reference frames has been described in:
    > [SHOT: Unique signatures of histograms for surface and texture
     description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    triangle_mesh: trimesh.Trimesh
        The triangle mesh on which we calculate the angles.
    distance_charts: np.array
        The geodesic distances calculated for the local chart.
    """
    angle_charts = []
    for origin_idx, chart in tqdm(
            enumerate(distance_charts),
            total=distance_charts.shape[0],
            desc="Computing angles using tangent plane projections"
    ):
        # Determine affine 3D neighborhood
        origin_vertex = triangle_mesh.vertices[origin_idx]
        neighborhood_indices = np.where(chart != np.inf)[0]
        chart_neighborhood = triangle_mesh.vertices[neighborhood_indices]
        chart_neighborhood = chart_neighborhood - origin_vertex

        # Determine weights for covariance matrix
        projections = get_2d_projections(chart_neighborhood)

        # Get angles
        angles = np.arctan2(projections[:, 1], projections[:, 0]) + np.pi

        # Assign angles
        angle_chart = np.full_like(chart, fill_value=-1.)
        angle_chart[neighborhood_indices] = angles
        angle_chart[origin_idx] = 0

        angle_charts.append(angle_chart)
    return np.array(angle_charts)
