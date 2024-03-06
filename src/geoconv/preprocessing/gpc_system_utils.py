from geoconv.utils.misc import get_faces_of_edge, compute_vector_angle

from scipy.linalg import blas

import c_extension
import numpy as np


def compute_u_ijk_and_angle(vertex_i, vertex_j, vertex_k, u, theta, object_mesh, use_c, rotation_axis):
    """Euclidean update procedure for a vertex i in a given triangle and angle computation

    See Section 3 in:

    > [Geodesic polar coordinates on polygonal meshes]
      (https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2012.03187.x)
    > Melvær, Eivind Lyche, and Martin Reimers.

    Parameters
    ----------
    vertex_i: int
        The index of the vertex for which we want to update the distance and angle
    vertex_j: int
        The index of the second vertex in the triangle of vertex i
    vertex_k: int
        The index of the third vertex in the triangle of vertex i
    u: np.ndarray
        The currently known radial coordinates
    theta: np.ndarray
        The currently known angular coordinates
    object_mesh: trimesh.Trimesh
        A loaded object mesh
    use_c: bool
        A flag whether to use the c-extension
    rotation_axis: np.ndarray [DEPRECATED]
        The vertex normal of the center vertex from the considered GPC-system

    Returns
    -------
    (float, float)
        The Euclidean update u_ijk for vertex i (see equation 13 in paper) and the new angle vertex i
    """

    # Convert indices to vectors
    u_j, u_k = u[[vertex_j, vertex_k]]
    theta_i_init, theta_j, theta_k = theta[[vertex_i, vertex_j, vertex_k]]
    vertex_i, vertex_j, vertex_k = object_mesh.vertices[[vertex_i, vertex_j, vertex_k]]

    if use_c:
        result = np.array([0., 0.])
        c_extension.compute_dist_and_dir(
            result,
            vertex_i,
            vertex_j,
            vertex_k,
            u_j,
            u_k,
            theta_j,
            theta_k,
            rotation_axis
        )
        u_ijk, theta_i = result
    else:
        e_j = np.empty(3)
        blas.dcopy(vertex_j, e_j)
        blas.daxpy(vertex_i, e_j, a=-1.0)
        e_j_norm = blas.dnrm2(e_j)

        e_k = np.empty(3)
        blas.dcopy(vertex_k, e_k)
        blas.daxpy(vertex_i, e_k, a=-1.0)
        e_k_norm = blas.dnrm2(e_k)

        e_kj = np.empty(3)
        blas.dcopy(vertex_k, e_kj)
        blas.daxpy(vertex_j, e_kj, a=-1.0)
        e_kj_sqnrm = blas.ddot(e_kj, e_kj)

        A = e_j_norm * e_k_norm * np.sin(compute_vector_angle(e_j, e_k, None))

        # variant 1:
        square_1, square_2 = np.square(np.array([u_j - u_k, u_j + u_k]))
        radicand = (e_kj_sqnrm - square_1) * (square_2 - e_kj_sqnrm)

        # variant 2:
        # c, b, a = np.sort(np.array([u_j, u_k, blas.dnrm2(e_kj)]))
        # radicand = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))

        if radicand <= 0:
            j = u_j + blas.dnrm2(e_j)
            k = u_k + blas.dnrm2(e_k)
            if j <= k:
                u_ijk = j
                theta_i = theta_j
            else:
                u_ijk = k
                theta_i = theta_k
        else:
            H = np.sqrt(radicand)
            u_j_sq, u_k_sq = np.square(np.array([u_j, u_k]))
            x_j = A * (e_kj_sqnrm + u_k_sq - u_j_sq) + blas.ddot(e_k, e_kj) * H
            x_k = A * (e_kj_sqnrm + u_j_sq - u_k_sq) - blas.ddot(e_j, e_kj) * H
            # If x_k < 0 or x_k < 0 then alpha > 1, causing theta_i to be negative (and we don't want that).
            if x_j < 0 or x_k < 0:
                j = u_j + blas.dnrm2(e_j)
                k = u_k + blas.dnrm2(e_k)
                if j <= k:
                    u_ijk = j
                    theta_i = theta_j
                else:
                    u_ijk = k
                    theta_i = theta_k
            else:
                # Compute distance
                denominator = 2 * A * e_kj_sqnrm
                x_j, x_k = np.array([x_j, x_k]) / denominator
                blas.dscal(x_j, e_j)
                blas.dscal(x_k, e_k)

                result_vector = np.empty(3)
                blas.dcopy(e_k, result_vector)
                blas.daxpy(e_j, result_vector)
                u_ijk = blas.dnrm2(result_vector)

                # Compute angle
                s = np.empty(3)
                blas.dcopy(result_vector, s)
                blas.daxpy(vertex_i, s)

                blas.daxpy(s, vertex_k, a=-1.0)
                blas.daxpy(s, vertex_j, a=-1.0)
                blas.daxpy(s, vertex_i, a=-1.0)

                phi_kj = compute_vector_angle(vertex_k, vertex_j, None)
                phi_ij = compute_vector_angle(vertex_i, vertex_j, None)
                alpha = phi_ij / phi_kj

                # Pay attention to 0-2pi-discontinuity
                if theta_k <= theta_j:
                    if theta_j - theta_k >= np.pi:
                        theta_k = theta_k + 2 * np.pi
                else:
                    if theta_k - theta_j >= np.pi:
                        theta_j = theta_j + 2 * np.pi
                theta_i = np.fmod((1 - alpha) * theta_j + alpha * theta_k, 2 * np.pi)

    return u_ijk, theta_i


def compute_distance_and_angle(vertex_i, vertex_j, gpc_system, use_c, rotation_axis):
    """Euclidean update procedure for geodesic distance approximation

    See Section 4 in:

    > [Geodesic polar coordinates on polygonal
    meshes](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2012.03187.x)
    > Melvær, Eivind Lyche, and Martin Reimers.

    Parameters
    ----------
    vertex_i: int
        The index of the vertex for which we want to update the distance and angle
    vertex_j: int
        The index of a second vertex in the triangle of vertex i (candidate vertex from heap)
    gpc_system: GPCSystem
        The current GPC-system.
    use_c: bool
        A flag whether to use the c-extension
    rotation_axis: np.ndarray [DEPRECATED]
        The vertex normal of the center vertex from the considered GPC-system

    Returns
    -------
    (float, float, list)
        The Euclidean update u_ijk for vertex i (see equation 13 in paper) and the new angle vertex i. Lastly, this
        function also returns the missing vertices which could have been used to update the coordinates of `vertex_i`.
    """
    # We consider both faces of `sorted_edge` for computing the coordinates to `vertex_i`
    sorted_edge = np.sort([vertex_i, vertex_j])
    if (sorted_edge[0], sorted_edge[1]) in gpc_system.faces.keys():
        # Use cache to get faces of `sorted_edge`
        considered_faces = gpc_system.faces[(sorted_edge[0], sorted_edge[1])]
    else:
        _, considered_faces = get_faces_of_edge(sorted_edge, gpc_system.object_mesh)

    # Compute GPC for `vertex_i` considering both faces of `[vertex_i, vertex_j]`
    updates = []
    k_vertices = []
    for face in considered_faces:
        vertex_k = [v for v in face if v not in [vertex_i, vertex_j]][0]
        k_vertices.append(vertex_k)
        # We need to know the distance to `vertex_k`
        if gpc_system.radial_coordinates[vertex_k] < np.inf and gpc_system.angular_coordinates[vertex_k] >= 0.:
            u_ijk, phi_i = compute_u_ijk_and_angle(
                vertex_i,
                vertex_j,
                vertex_k,
                gpc_system.radial_coordinates,
                gpc_system.angular_coordinates,
                gpc_system.object_mesh,
                use_c,
                rotation_axis
            )
            updates.append((u_ijk, phi_i, vertex_k))

    # If no GPC have been found for `vertex_i`, return default GPC
    if not updates:
        return np.inf, -1.0, None

    # If two GPC have been found for `vertex_i`, return the smallest distance to `vertex_i`
    else:
        u_ijk, phi_i, vertex_k = min(updates)
        return u_ijk, phi_i, k_vertices
