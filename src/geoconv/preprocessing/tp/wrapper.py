from geoconv.preprocessing.tp.angles import compute_angles_for_projections
from geoconv.preprocessing.tp.tangent_projections import get_3d_neighborhood, get_2d_projections

import numpy as np


def pickable_tp(idx_subset, vertices, eucl_max_radius):
    """Computes local charts using only tangent plane projections.

    Parameters
    ----------
    idx_subset: np.ndarray
        Array of indices for the vertices around which charts should be computed.
    vertices: np.ndarray
        The 3D coordinates of the shape.
    eucl_max_radius: float
        The maximum radius for neighbors to consider for tangent projections.

    Returns
    -------
    np.ndarray:
        All local charts for the given shape vertices.
    """
    # Get Euclidean 3D neighborhoods
    neighborhoods, neighborhood_indices = get_3d_neighborhood(
        vertices, max_radius=eucl_max_radius, return_as_array=False, required_origins=idx_subset
    )

    # Get 2D projections
    all_projections = [get_2d_projections(np.array(hood)) for hood in neighborhoods]

    # Compute charts
    angles = [compute_angles_for_projections(p) for p in all_projections]
    distances = [np.linalg.norm(p, axis=-1) for p in all_projections]
    charts = [np.stack([d, a], axis=-1) for (d, a) in zip(distances, angles)]

    # Adjust charts into correct shape
    n_origins, n_vertices = idx_subset.shape[0], vertices.shape[0]
    padded_charts = np.stack(
        [np.full((n_origins, n_vertices), np.inf), np.full((n_origins, n_vertices), -1.)],
        axis=-1
    )
    padded_charts[neighborhood_indices] = np.array([coord for chart in charts for coord in chart])
    return padded_charts
