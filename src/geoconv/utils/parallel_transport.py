from tqdm import tqdm

import potpourri3d as pp3d
import numpy as np


def compute_parallel_transport(triangle_mesh, chart_indices=None):
    """Computes the parallel transport of x-axes between all pairs of charts of a surface.

    This function uses the Vector Heat method to compute parallel transports:
    > [The Vector Heat Method](https://dl.acm.org/doi/10.1145/3243651)
    > Nicolas Sharp, Yousuf Soliman and Keenan Crane

    Implementation for the Vector Heat method available at:
    > https://github.com/nmwsharp/potpourri3d

    Parameters
    ----------
    triangle_mesh: trimesh.Trimesh
        The triangle mesh for whose vertex pairs parallel transports shall be computed.
    chart_indices: np.ndarray | None
        The indices of origin vertices for which parallel transports should be calculated.
        If 'None', all origin vertices are used.

    Return
    ------
    np.ndarray:
        A symmetric (N x N) matrix, where N equals the amount of vertices of 'triangle_mesh'. Entry [a, b] contains the
        rotation angle that the x-axis in the local chart at vertex 'a' has to be rotated with to be represented in the
        local chart at vertex 'b' (and vice versa because of matrix symmetry). If 'chart_indices' are provided, the
        matrix reduces to (len(chart_indices) x N).
    """
    solver = pp3d.MeshVectorHeatSolver(V=triangle_mesh.vertices, F=triangle_mesh.faces)
    transport_vector = np.array([1., 0.])  # transport x-axis
    angles_n_x_n = []
    if chart_indices is None:
        chart_indices = range(triangle_mesh.vertices.shape[0])
    for idx in tqdm(chart_indices, desc="Computing parallel transport..."):
        result = solver.transport_tangent_vector(v_ind=idx, vector=transport_vector)
        angles = np.arccos(np.einsum("i,ni->n", transport_vector, result))
        angles_n_x_n.append(angles)
    return np.array(angles_n_x_n)


def concat_bc_and_angles(bc, angles):
    """Retrieves the required angles for given barycentric coordinates and concatenates angles onto last dim. of 'bc'.

    Parameters
    ----------
    bc: np.ndarray
        The barycentric coordinates,
    angles: np.ndarray
        The (n_vertices x n_vertices) matrix containing the angles required for parallel transport.

    Returns
    -------
    np.ndarray:
        An array 'bc' that contains both the barycentric coordinates and their required angles for the parallel
        transport. It has shape: (batch, n_vertices, n_radial, n_angular, 3, 3). Thereby, bc[..., 2] contains the
        angles for vertex index bc[..., 1].
    """
    # Get indices of barycentric coordinates
    # 'bc_indices': (batch, n_vertices, n_radial, n_angular, 3)
    # _, bc_indices = tf.unstack(bc, axis=-1)
    bc_indices = bc[..., 1].astype(np.int32)

    # Gather angles
    angles = angles[np.arange(bc.shape[0])[:, None, None, None], bc_indices]

    # Concat bc and angles
    return np.concatenate([bc, angles[..., None]], axis=-1)
