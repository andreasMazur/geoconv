from scipy.linalg import blas
from tqdm import tqdm

import numpy as np
import c_extension
import heapq
import warnings


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
    (float, float, dict)
        The Euclidean update u_ijk for vertex i (see equation 13 in paper) and the new angle vertex i. Also, the
        possibly updated face cache is returned.
    """

    if face_cache is None:
        face_cache = dict()

    # Caching considered triangles to save time
    sorted_edge = (vertex_i if vertex_i < vertex_j else vertex_j, vertex_j if vertex_j > vertex_i else vertex_i)
    if sorted_edge in face_cache.keys():
        considered_faces = face_cache[sorted_edge]
    else:
        # Determine both triangles of 'sorted_edge' and cache those
        edge_indices = object_mesh.edges_sorted == sorted_edge
        edge_indices = np.where(np.logical_and(edge_indices[:, 0], edge_indices[:, 1]))

        face_indices = object_mesh.edges_face[edge_indices]
        considered_faces = object_mesh.faces[face_indices]

        face_cache[sorted_edge] = considered_faces

    updates = []
    for triangle in considered_faces:
        vertex_k = [v for v in triangle if v not in [vertex_i, vertex_j]][0]
        # We need to know the distance to `vertex_k`
        if u[vertex_k] < np.inf and theta[vertex_k] >= 0.:
            u_ijk, phi_i = compute_u_ijk_and_angle(
                vertex_i, vertex_j, vertex_k, u, theta, object_mesh, use_c, rotation_axis
            )
            updates.append((u_ijk, phi_i))
    if not updates:
        return np.inf, -1.0, face_cache
    else:
        u_ijk, phi_i = min(updates)
        return u_ijk, phi_i, face_cache


def compute_vector_angle(vector_a, vector_b, rotation_axis):
    """Compute the angle between two vectors

    Parameters
    ----------
    vector_a: np.ndarray
        The first vector
    vector_b: np.ndarray
        The second vector
    rotation_axis: [np.ndarray, None]
        For angles in [0, 2*pi[ in the 3-dimensional space an "up"-direction is required. If `None` is passed an angle
        between [0, pi[ is returned.

    Returns
    -------
    float:
        The angle between `vector_a` and `vector_b`
    """
    vector_a = vector_a / blas.dnrm2(vector_a)
    vector_b = vector_b / blas.dnrm2(vector_b)
    angle = blas.ddot(vector_a, vector_b)
    if angle > 1.0:
        angle = 1.0
    elif angle < -1.0:
        angle = -1.0
    angle = np.arccos(angle)
    if rotation_axis is None:
        return angle
    else:
        cross_product = np.cross(vector_a, vector_b)
        opposite_direction = rotation_axis.dot(cross_product) < 0.0
        angle = 2 * np.pi - angle if opposite_direction else angle
        return angle


def get_neighbors(vertex, object_mesh):
    """Calculates the one-hop neighbors of a vertex

    Parameters
    ----------
    vertex: int
        The index of the vertex for which the neighbor indices shall be computed
    object_mesh: trimesh.Trimesh
        An object mesh

    Returns
    -------
    list:
        A list of neighboring vertex-indices.
    """

    return list(object_mesh.vertex_adjacency_graph[vertex].keys())


def initialize_neighborhood(source_point, u, theta, object_mesh, use_c):
    """Compute the initial radial and angular coordinates around a source point.

    Angle coordinates are always given w.r.t. some reference direction. The choice of a reference
    direction can be arbitrary. Here, we choose the vector `x - source_point` with `x` being the
    first neighbor return by `get_neighbors` as the reference direction.

    Parameters
    ----------
    source_point: int
        The index of the source point around which a window (GPC-system) shall be established
    u: np.ndarray
        An array `u` of radial coordinates from the source point to other points in the object mesh
    theta: np.ndarray
        An array `theta` of angular coordinates of neighbors from `source_point` in its window
    object_mesh: trimesh.Trimesh
        A loaded object mesh
    use_c: bool
        A flag whether to use the c-extension

    Returns
    -------
    (np.ndarray, np.ndarray, list, np.ndarray):
        This function returns updated radial coordinates `u` (fst. value), updated angular coordinates `theta` (snd.
        value), The neighbors of `source_point` (thr. value) and lastly the rotation axis for this GPC-system (vertex-
        normal of the center vertex).
    """

    source_point_neighbors = get_neighbors(source_point, object_mesh)
    r3_source_point = object_mesh.vertices[source_point]
    ref_neighbor = source_point_neighbors[0]

    # Calculate neighbor values: radial coordinates
    r3_neighbors = object_mesh.vertices[source_point_neighbors]
    u[source_point_neighbors] = np.linalg.norm(
        r3_neighbors - np.stack([r3_source_point for _ in range(len(source_point_neighbors))]), ord=2, axis=-1
    )

    # Calculate neighbor values: angular coordinates
    rotation_axis = object_mesh.vertex_normals[source_point]
    theta_neighbors = np.full((len(source_point_neighbors,)), .0)
    for idx, neighbor in enumerate(source_point_neighbors):
        vector_a = object_mesh.vertices[ref_neighbor] - object_mesh.vertices[source_point]
        vector_b = object_mesh.vertices[neighbor] - object_mesh.vertices[source_point]
        if use_c:
            theta_neighbors[idx] = c_extension.compute_angle_360(vector_a, vector_b, rotation_axis)
        else:
            theta_neighbors[idx] = compute_vector_angle(vector_a, vector_b, rotation_axis)

    theta[source_point_neighbors] = theta_neighbors
    theta[source_point] = 0.0

    return u, theta, source_point_neighbors, rotation_axis


def local_gpc(source_point, u_max, object_mesh, use_c, eps=0.000001, triangle_cache=None):
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
    triangle_cache: dict
        A cache storing the faces of a given edge

    Returns
    -------
    (np.ndarray, np.ndarray, dict)
        An array `u` of radial coordinates from the source point to other points in the object mesh. An array `theta` of
        angular coordinates of neighbors from `source_point` in its window. The possibly updated triangle cache, which
        associates seen triangles to edges, is also returned as a dictionary.
    """

    # Initialization
    u = np.full((object_mesh.vertices.shape[0],), np.inf)
    theta = np.full((object_mesh.vertices.shape[0],), -1.0)
    u, theta, source_point_neighbors, rotation_axis = initialize_neighborhood(
        source_point, u, theta, object_mesh, use_c
    )
    u[source_point] = .0

    check_array = np.array([x for x in u if not np.isinf(x)])
    if check_array.max() > u_max:
        warnings.warn(
            f"You chose a 'u_max' to be smaller then {check_array.max()}, which has been seen as an initialization"
            f" length for a GPC-system. Current GPC-system will only contain initialization vertices.", RuntimeWarning
        )

    candidates = []  # heap containing vertices sorted by distance u[i]
    for neighbor in source_point_neighbors:
        candidates.append((u[neighbor], neighbor))
    heapq.heapify(candidates)

    while candidates:
        # as we work with a min-heap the shortest distance is stored in root
        j_dist, j = heapq.heappop(candidates)
        j_neighbors = get_neighbors(j, object_mesh)
        j_neighbors = [j for j in j_neighbors if j != source_point]

        for i in j_neighbors:
            # during computation of distance to `i` consider both triangles which contain (i, j)
            # [compare section 4.1 and algorithm 1]
            new_u_i, new_theta_i, triangle_cache = compute_distance_and_angle(
                i, j, u, theta, triangle_cache, object_mesh, use_c, rotation_axis
            )

            # In difference to the original pseudocode, we add 'new_u_i < u_max' to this IF-query
            # to ensure that the radial coordinates do not exceed 'u_max'.
            if new_u_i < np.inf and u[i] / new_u_i > 1 + eps and new_u_i < u_max:
                u[i] = new_u_i
                theta[i] = new_theta_i
                # if new_u_i < u_max:
                heapq.heappush(candidates, (new_u_i, i))

    return u, theta, triangle_cache


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
    triangle_cache = dict()
    if tqdm_msg:
        for vertex_idx in tqdm(range(object_mesh.vertices.shape[0]), position=0, postfix=tqdm_msg):
            u_v, theta_v, triangle_cache = local_gpc(
                vertex_idx, u_max, object_mesh, use_c, eps, triangle_cache
            )
            gpc_systems.append(np.stack([u_v, theta_v], axis=1))
    else:
        for vertex_idx in range(object_mesh.vertices.shape[0]):
            u_v, theta_v, triangle_cache = local_gpc(
                vertex_idx, u_max, object_mesh, use_c, eps, triangle_cache
            )
            gpc_systems.append(np.stack([u_v, theta_v], axis=1))

    return np.stack(gpc_systems)
