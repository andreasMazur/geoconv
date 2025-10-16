from tqdm import tqdm

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
        neighbor_distances = np.linalg.norm(chart_neighborhood, axis=-1)
        chart_neighborhood_radius = np.linalg.norm(chart_neighborhood, axis=-1).max()
        neighbor_weights = chart_neighborhood_radius - neighbor_distances

        # Compute distance-weighted covariance matrix
        cov_matrix = 1 / neighbor_weights.sum() * np.einsum(
            "n,ni,nj->ij", neighbor_weights, chart_neighborhood, chart_neighborhood
        )

        # Get eigenvalues and -vectors and disambiguate axes according to SHOT-paper
        # Smallest eigenvalue: eigenvalues[0]
        # Associated eigenvector: eigenvectors[:, 0]
        _, eigenvectors = np.linalg.eigh(cov_matrix)
        z_axis = disambiguate_axis(eigenvectors[:, 0], chart_neighborhood)  # (associated to smallest eigenvalue)
        x_axis = disambiguate_axis(eigenvectors[:, 2], chart_neighborhood)  # (associated to largest eigenvalue)
        y_axis = np.cross(z_axis, x_axis)

        # Project neighborhood into tangent plane (plane spanned by x- and y-axis)
        projections = chart_neighborhood - np.einsum("c,nc->n", z_axis, chart_neighborhood)[:, None] * z_axis[None, :]

        # Basis change (first dim is zero, as we have projected into plane of z-axis)
        projections = np.einsum(
            "ij,nj->ni", np.linalg.inv(np.array([z_axis, y_axis, x_axis]).T), projections
        )[:, 1:]

        # Get angles
        angles = np.arctan2(projections[:, 1], projections[:, 0]) * 180 / np.pi + 180

        # Assign angles
        angle_chart = np.full_like(chart, fill_value=-1.)
        angle_chart[neighborhood_indices] = angles
        angle_chart[origin_idx] = 0

        angle_charts.append(angle_chart)
    return np.array(angle_charts)
