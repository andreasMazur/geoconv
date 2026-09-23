import potpourri3d as pp3d
import numpy as np


def pickable_fmm(vertex_indices, vertices, faces):
    """Wrapper for the Fast Marching Method for triangle meshes provided by the Potpourri3D library.

    The fast marching method has been published in:
    > [A fast marching level set method for monotonically advancing fronts](www.pnas.org/doi/pdf/10.1073/pnas.93.4.1591)
    > James A. Sethian

    The fast marching method for triangle meshes has been published in:
    > [Computing geodesic paths on manifolds](https://www.pnas.org/doi/pdf/10.1073/pnas.95.15.8431)
    > Ron Kimmel and James A. Sethian

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
    solver = pp3d.MeshFastMarchingDistanceSolver(V=vertices, F=faces)
    index_distances = np.array([solver.compute_distance([[(v_idx, [])]], sign=False) for v_idx in vertex_indices])
    return index_distances
