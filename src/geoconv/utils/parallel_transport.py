from tqdm import tqdm

import potpourri3d as pp3d
import numpy as np


def compute_parallel_transport(triangle_mesh, gc_x_axes, source_vertex_indices=None):
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
    gc_x_axes: np.ndarray
        The x-axes of the computed surface charts in 3D coordinates.
    source_vertex_indices: np.ndarray | None
        The indices of the source vertices for which parallel transports are required.

    Return
    ------
    np.ndarray:
        A (N x N) matrix, where N equals the amount of vertices of 'triangle_mesh'. Entry [a, b] contains the
        rotation angle that the x-axis in the local chart at vertex 'a' has to be rotated with to be represented in the
        local chart at vertex 'b'.
    """
    # Initialize solver
    solver = pp3d.MeshVectorHeatSolver(V=triangle_mesh.vertices, F=triangle_mesh.faces)

    # 3D LRFs from potpourri3d
    pp_x_axes, pp_y_axes, _ = solver.get_tangent_frames()
    pp_x_axes = pp_x_axes / np.linalg.norm(pp_x_axes, axis=-1)[:, None]
    pp_y_axes = pp_y_axes / np.linalg.norm(pp_y_axes, axis=-1)[:, None]

    # 3D x-axes of LRFs from GeoConv, Atlas-class
    x_axes_norm = np.linalg.norm(gc_x_axes, axis=-1)
    no_neighbors_mask = x_axes_norm == 0.
    gc_x_axes[np.logical_not(no_neighbors_mask)] = (
            gc_x_axes[np.logical_not(no_neighbors_mask)] / x_axes_norm[np.logical_not(no_neighbors_mask), None]
    )

    # Replace non-existent x-axis with those of potpourri3d
    gc_x_axes[no_neighbors_mask] = pp_x_axes[no_neighbors_mask]

    # Signed angle offsets between GeoConv and potpourri3d frames
    # 'correction_angles' stores angles theta by which PP-x-axes need to be rotated with to coincide with GC's x-axes.
    # Thus, 'correction_angles' stores angles that cause a basis change PP -> GC.
    correction_angles = np.arctan2(
        # Dot product with basis vector yields coordinate in basis direction.
        # Represents GC x-axes in potpourri3d's coordinate frames.
        np.einsum("vi,vi->v", gc_x_axes, pp_y_axes),
        np.einsum("vi,vi->v", gc_x_axes, pp_x_axes)
    )

    # Correction angle of PP -> PP is zero (machine accuracy sometimes comes to its limits here)
    correction_angles[no_neighbors_mask] = 0.

    # Transport x-axis
    transport_vector = np.array([1., 0.])

    # All parallel transport angles
    angles_n_x_n = []

    # The indices of the charts, origin of the parallel transports
    if source_vertex_indices is None:
        source_vertex_indices = np.arange(triangle_mesh.vertices.shape[0])

    # Compute the parallel transports
    for idx in tqdm(source_vertex_indices, desc="Computing parallel transport..."):
        result = solver.transport_tangent_vector(v_ind=idx, vector=transport_vector)

        # Compute transport angles using the Vector Heat Method
        transport_angles = np.arctan2(result[:, 1], result[:, 0])

        # Angle correction due to change of basis between GeoConv and potpourri3d
        # Difference angle in origin: GC_1 -> P_1, "-correction_angles[idx]"
        # Transport angle: P_1 -> P_2, transport_angles
        # Difference angle in target vertex: P_2 -> GC_2, "+correction_angles"
        angles = np.mod(-correction_angles[idx] + transport_angles + correction_angles, 2 * np.pi)
        angles[np.isclose(angles, 2 * np.pi)] = 0.

        angles_n_x_n.append(angles)

    # Return the parallel transport angles
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
