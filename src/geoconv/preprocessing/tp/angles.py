from geoconv.preprocessing.tp.tangent_projections import get_2d_projections

from tqdm import tqdm

import numpy as np


def compute_angles_for_distances(triangle_mesh, local_geodesic_dists):
    """Computes the angles for given local charts via tangent plane projections.

    Computation of reference frames has been described in:
    > [SHOT: Unique signatures of histograms for surface and texture
     description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    triangle_mesh: trimesh.Trimesh
        The triangle mesh on which we calculate the angles.
    local_geodesic_dists: np.array
        The geodesic distances calculated for the local chart.
    """
    angle_charts = []
    for origin_idx, chart in tqdm(
            enumerate(local_geodesic_dists),
            total=local_geodesic_dists.shape[0],
            desc="Computing angles using tangent plane projections"
    ):
        # Determine affine 3D neighborhood
        origin_vertex = triangle_mesh.vertices[origin_idx]
        neighborhood_indices = np.where(chart != np.inf)[0]
        chart_neighborhood = triangle_mesh.vertices[neighborhood_indices]
        chart_neighborhood = chart_neighborhood - origin_vertex

        # Determine weights for covariance matrix
        projections = get_2d_projections(chart_neighborhood, rescale=False)

        # Get angles
        angles = np.arctan2(projections[:, 1], projections[:, 0]) + np.pi

        # Assign angles
        angle_chart = np.full_like(chart, fill_value=-1.)
        angle_chart[neighborhood_indices] = angles
        angle_chart[origin_idx] = 0

        angle_charts.append(angle_chart)
    return np.array(angle_charts)


def compute_angles_for_projections(projections):
    """

    Parameters
    ----------
    projections: np.ndarray
        The XY-coordinates of the tangent plane projections.

    Returns
    -------
    np.ndarray:
        The angles for all tangent plane projections.
    """
    angles = np.arctan2(projections[..., 1], projections[..., 0])
    not_origin = (projections != [0., 0.]).all(axis=-1)
    angles[not_origin] = angles[not_origin] + np.pi
    return angles
