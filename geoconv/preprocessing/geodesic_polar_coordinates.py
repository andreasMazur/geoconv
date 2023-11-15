from geoconv.preprocessing.gpc_system import GPCSystem
from geoconv.utils.misc import get_neighbors, get_faces_of_edge, compute_vector_angle

from scipy.linalg import blas
from tqdm import tqdm

import c_extension
import numpy as np
import warnings
import heapq


def compute_u_ijk_and_angle(vertex_i, vertex_j, vertex_k, u, theta, object_mesh, use_c, rotation_axis):
    """Euclidean update procedure for a vertex i in a given triangle and angle computation

    See Section 3 in:

    > [Geodesic polar coordinates on polygonal
    meshes](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2012.03187.x)
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


def compute_distance_and_angle(vertex_i, vertex_j, u, theta, face_cache, object_mesh, use_c, rotation_axis):
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
    u: np.ndarray
        The currently known radial coordinates
    theta: np.ndarray
        The currently known angular coordinates
    face_cache: dict
        A cache storing (or not storing) the faces for edge `(vertex_i, vertex_j)`
    object_mesh: trimesh.Trimesh
        A loaded object mesh
    use_c: bool
        A flag whether to use the c-extension
    rotation_axis: np.ndarray [DEPRECATED]
        The vertex normal of the center vertex from the considered GPC-system

    Returns
    -------
    (float, float, int)
        The Euclidean update u_ijk for vertex i (see equation 13 in paper) and the new angle vertex i. Lastly, this
        function also returns the missing vertex that is used to update the coordinates of `vertex_i`.
    """
    # We consider both faces of `sorted_edge` for computing the coordinates to `vertex_i`
    sorted_edge = np.sort([vertex_i, vertex_j])
    if (sorted_edge[0], sorted_edge[1]) in face_cache.keys():
        # Use cache to get faces of `sorted_edge`
        considered_faces = face_cache[(sorted_edge[0], sorted_edge[1])]
    else:
        # Cache considered faces to save time
        _, considered_faces = get_faces_of_edge(sorted_edge, object_mesh)
        face_cache[(sorted_edge[0], sorted_edge[1])] = considered_faces

    # Compute GPC for `vertex_i` considering both faces of `[vertex_i, vertex_j]`
    updates = []
    for face in considered_faces:
        vertex_k = [v for v in face if v not in [vertex_i, vertex_j]][0]
        # We need to know the distance to `vertex_k`
        if u[vertex_k] < np.inf and theta[vertex_k] >= 0.:
            u_ijk, phi_i = compute_u_ijk_and_angle(
                vertex_i, vertex_j, vertex_k, u, theta, object_mesh, use_c, rotation_axis
            )
            updates.append((u_ijk, phi_i, vertex_k))

    # If no GPC have been found for `vertex_i`, return default GPC
    if not updates:
        return np.inf, -1.0, None
    # If two GPC have been found for `vertex_i`, return the smallest distance to `vertex_i`
    else:
        u_ijk, phi_i, vertex_k = min(updates)
        return u_ijk, phi_i, vertex_k


def compute_gpc_system(source_point, u_max, object_mesh, use_c, eps=0.000001, face_cache=None, visualize=False):
    """Computes local GPC for one given source point.

    Parameters
    ----------
    source_point: int
        The index of the source point around which a window (GPC-system) shall be established
    u_max: float
        The maximal distance (e.g. radius of the patch) which a vertex may have to `source_point`
    object_mesh: trimesh.Trimesh
        A loaded object mesh
    use_c: bool
        A flag whether to use the c-extension
    eps: float
        An epsilon
    face_cache: dict
        A cache storing the faces of a given edge
    visualize: bool
        Whether to visualize the process of creating the GPC-system

    Returns
    -------
    (np.ndarray, np.ndarray, dict, dict, dict)
        An array `u` of radial coordinates from the source point to other points in the object mesh. An array `theta` of
        angular coordinates of neighbors from `source_point` in its window. The possibly updated face cache, which
        associates seen faces to edges, is returned as a dictionary at the third position. The second to last dictionary
        contains all contained edges within the gpc-system. The last dictionary contains all faces which are contained
        within the GPC-system, i.e. which consist of nodes that have geodesic polar coordinates.
    """

    #######################################
    # Initialize GPC-system and face-cache
    #######################################
    gpc_system = GPCSystem(source_point, object_mesh, faces=face_cache, use_c=True)
    # Check whether initialization distances are larger than given max-radius
    check_array = np.array([x for x in gpc_system.radial_coordinates if not np.isinf(x)])
    if check_array.max() > u_max:
        warnings.warn(
            f"You chose a 'u_max' to be smaller then {check_array.max()}, which has been seen as an initialization"
            f" length for a GPC-system. Current GPC-system will only contain initialization vertices.",
            RuntimeWarning
        )

    ############################################
    # Initialize min-heap over radial distances
    ############################################
    candidates = []
    for neighbor in source_point_neighbors:
        candidates.append((gpc_system.radial_coordinates[neighbor], neighbor))
    heapq.heapify(candidates)

    ###################################
    # Algorithm to compute GPC-systems
    ###################################
    while candidates:
        # Get vertex from min-heap that is closest to GPC-system origin
        j_dist, j = heapq.heappop(candidates)
        j_neighbors = get_neighbors(j, object_mesh)
        j_neighbors = [j for j in j_neighbors if j != source_point]
        for i in j_neighbors:
            # Compute the (updated) geodesic distance `new_u_i` and angular coordinate of the i-th neighbor from the
            # closest vertex in the min-heap to the source point of the GPC-system
            new_u_i, new_theta_i, k = compute_distance_and_angle(
                i, j, u, theta, face_cache, object_mesh, use_c, rotation_axis
            )

            # In difference to the original pseudocode, we add 'new_u_i < u_max' to this IF-query
            # to ensure that the radial coordinates do not exceed 'u_max'.
            if new_u_i < u_max and u[i] / new_u_i > 1 + eps:
                if update_edges(i, new_u_i, new_theta_i, j, k, u, theta, edge_cache, visualize):
                    u[i] = new_u_i
                    theta[i] = new_theta_i
                    heapq.heappush(candidates, (new_u_i, i))

    # Collect all faces that are entirely described in the GPC-system
    gpc_faces = []
    for edge in edge_cache[-1]:
        faces_of_edge = face_cache[(edge[0], edge[1])]
        for face in faces_of_edge:
            face = list(face)
            if np.inf not in u[face] and face not in gpc_faces:
                gpc_faces.append(face)

    return u, theta, face_cache, edge_cache, gpc_faces


def compute_gpc_systems(object_mesh, u_max=.04, eps=0.000001, use_c=True, tqdm_msg=""):
    """Computes approximated geodesic polar coordinates for all vertices within an object mesh.

    > [Geodesic polar coordinates on polygonal
    meshes](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2012.03187.x)
    > Melvær, Eivind Lyche, and Martin Reimers.

    Parameters
    ----------
    object_mesh: trimesh.Trimesh
        An object mesh
    u_max: float
        The maximal radius for the GPC-system
    eps: float
        A threshold for update-improvements
    use_c: bool
        A flag whether to use the c-extension
    tqdm_msg: str
        A string to display as suffix with tqdm

    Returns
    -------
    np.ndarray:
        Array A with dimensions `(n, n, 2)` with `n = object_mesh.vertices.shape[0]`. A[i][j][0] stores the radial
        distance (with max value `u_max`) from node `j` to origin `i` of the local GPC-system. A[i][j][1] contains
        the radial coordinate of node `j` in the local GPC-system of node `i` w.r.t. a reference direction (see
        `initialize_neighborhood` for how the reference direction is selected).
    """

    gpc_systems = []
    face_cache = dict()
    if tqdm_msg:
        for vertex_idx in tqdm(range(object_mesh.vertices.shape[0]), position=0, postfix=tqdm_msg):
            u_v, theta_v, face_cache, edge_cache, gpc_faces = compute_gpc_system(
                vertex_idx, u_max, object_mesh, use_c, eps, face_cache, visualize=False
            )
            gpc_systems.append(np.stack([u_v, theta_v], axis=1))
    else:
        for vertex_idx in range(object_mesh.vertices.shape[0]):
            u_v, theta_v, face_cache, edge_cache, gpc_faces = compute_gpc_system(
                vertex_idx, u_max, object_mesh, use_c, eps, face_cache, visualize=False
            )
            gpc_systems.append(np.stack([u_v, theta_v], axis=1))
    return np.stack(gpc_systems)
