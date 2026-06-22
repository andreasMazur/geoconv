from geoconv.preprocessing.dgpc.dgpc_solver import DgpcSolver
from tqdm import tqdm

import numpy as np


def pickable_dgpc(vertex_indices, vertices, faces, u_max):
    """A pickable wrapper for the DGPC-algorithm.

    Parameters
    ----------
    vertex_indices: list
        A list of vertex indices that the DGPC solver will use as source points.
    vertices: np.ndarray
        The mesh vertices.
    faces: np.ndarray
        The mesh faces. I.e., the indices of vertex triangles put together into triples.
    u_max: float
        The maximum allowed geodesic distance originating from any source point in 'vertex_indices'.

    Returns
    -------
    np.ndarray:
        The local charts around 'vertex_indices'.
    """
    solver = DgpcSolver(V=vertices, F=faces, u_max=u_max)
    gpc_systems = []
    for source_point in tqdm(vertex_indices):
        gpc_systems.append(np.stack(solver.compute_dgpc(source_point), axis=-1))
    gpc_systems = np.array(gpc_systems)
    # gpc_systems = np.array([np.stack(solver.compute_dgpc(source_point), axis=-1) for source_point in vertex_indices])
    return gpc_systems
