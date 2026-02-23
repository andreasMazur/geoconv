from geoconv.preprocessing.tangent_proj.angles import compute_angles_for_projections
from geoconv.preprocessing.tangent_proj.project import get_3d_neighborhood, get_2d_projections

import numpy as np


def tangent_proj_chart(vertices, eucl_max_neighbors):
    """Computes local charts using only tangent plane projections.

    Parameters
    ----------
    vertices: np.ndarray
        The 3D coordinates of the shape.
    eucl_max_neighbors: int
        The maximum number of neighbors to consider for tangent projections.

    Returns
    -------
    np.ndarray:
        All local charts for the given shape vertices.
    """
    # Get Euclidean 3D neighborhoods
    neighborhoods_, indices_ = get_3d_neighborhood(vertices, max_neighbors=eucl_max_neighbors)

    # Get 2D projections
    all_projections_ = np.array([get_2d_projections(n_) for n_ in neighborhoods_])

    # Compute charts
    angles = compute_angles_for_projections(all_projections_)
    distances = np.linalg.norm(all_projections_, axis=-1)
    charts = np.stack([distances, angles], axis=-1)

    # Adjust charts into correct shape
    n_vertices = vertices.shape[0]
    padded_charts = np.stack(
        [np.full((n_vertices, n_vertices), np.inf), np.full((n_vertices, n_vertices), -1.)],
        axis=-1
    )
    padded_charts[np.arange(n_vertices)[:, None], indices_] = charts
    return padded_charts
