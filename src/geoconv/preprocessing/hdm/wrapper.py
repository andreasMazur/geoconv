import potpourri3d as pp3d
import numpy as np


def pickable_hdm(vertex_indices, vertices, faces):
    """Wrapper for the Heat Diffusion Method for Distance Calculations from Potpourri3D.

    The heat diffusion for distance calculation has been published in:
    > [The heat method for distance computation](https://dl.acm.org/doi/10.1145/3131280)
    > Keenan Crane, Clarisse Weischedel, Max Wardetzky

    Code available on GitHub:
    > https://github.com/nmwsharp/potpourri3d

    Parameters
    ----------
    vertex_indices: np.ndarray
        The vertex indices to compute distances for.
    vertices: np.ndarray
        All vertices of the mesh.
    faces: np.ndarray
        All faces of the mesh.

    Returns
    -------
    np.ndarray
        The computed distances for the given vertex indices.
    """
    solver = pp3d.MeshHeatMethodDistanceSolver(V=vertices, F=faces)
    index_distances = np.array([solver.compute_distance(v_idx) for v_idx in vertex_indices])
    return index_distances
