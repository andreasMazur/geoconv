from tqdm import tqdm

import potpourri3d as pp3d
import numpy as np


def compute_parallel_transport(triangle_mesh):
    """Computes rotation angles via parallel transport between all charts of a surface.

    This function uses the Vector Heat method to compute parallel transports:
    > [The Vector Heat Method](https://dl.acm.org/doi/10.1145/3243651)
    > Nicolas Sharp, Yousuf Soliman and Keenan Crane

    Implementation for the Vector Heat method available at:
    > https://github.com/nmwsharp/potpourri3d

    Parameters
    ----------
    triangle_mesh: trimesh.Trimesh
        The triangle mesh for whose vertex pairs parallel transports shall be computed.

    Return
    ------
    np.ndarray:
        A (N x N) matrix, where N equals the amount of vertices of 'triangle_mesh'. Entry [a, b] contains the rotation
        angle that a tangent vector in the tangent plane of vertex 'a' has to be rotated with to be represented in
        the tangent plane at vertex 'b'.
    """
    solver = pp3d.MeshVectorHeatSolver(V=triangle_mesh.vertices, F=triangle_mesh.faces)
    transport_vector = np.array([0., 1.])
    angles_n_x_n = []
    for idx in tqdm(range(triangle_mesh.vertices.shape[0]), desc="Computing parallel transport..."):
        result = solver.transport_tangent_vector(v_ind=idx, vector=transport_vector)
        angles = np.arccos(np.einsum("i,ni->n", transport_vector, result))
        angles_n_x_n.append(angles)
    return np.array(angles_n_x_n)
