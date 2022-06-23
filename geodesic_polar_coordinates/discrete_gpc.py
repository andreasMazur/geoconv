from scipy.linalg import blas

import numpy as np
import networkx as nx
import c_extension
import heapq
import tqdm
import sys


def compute_u_ijk_and_angle(vertex_i, vertex_j, vertex_k, u, theta, object_mesh, use_c, rotation_axis):
    """Euclidean update procedure for a vertex i in a given triangle and angle computation.

    See Section 3 in:

    > [Geodesic polar coordinates on polygonal
    meshes](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2012.03187.x)
    > Melvær, Eivind Lyche, and Martin Reimers.

    **Input**

    - The index of the vertex i for which we want to update the distance.
    - The index of the second vertex j in the triangle of vertex i.
    - The index of the third vertex k in the triangle of vertex i.
    - The currently known radial coordinates `u`.
    - The currently known angular coordinates `theta`.
    - A loaded object mesh.
    - A flag whether to use the c-extension.

    **Output**

    - The Euclidean update u_ijk (see equation 13 in paper)

    """

    # Convert indices to vectors
    u_j, u_k = u[[vertex_j, vertex_k]]
    theta_j, theta_k = theta[[vertex_j, vertex_k]]
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
                phi_kj = compute_vector_angle(vertex_k, vertex_j, rotation_axis)
                phi_ij = compute_vector_angle(vertex_i, vertex_j, rotation_axis)
                if not phi_kj:
                    j = u_j + blas.dnrm2(e_j)
                    k = u_k + blas.dnrm2(e_k)
                    if j <= k:
                        u_ijk = j
                        theta_i = theta_j
                    else:
                        u_ijk = k
                        theta_i = theta_k
                else:
                    # TODO: Why can `phi_ij` be greater than `phi_kj`?
                    if phi_ij < phi_kj:
                        alpha = phi_ij / phi_kj
                        if theta_j == 0 and theta_k > np.pi:
                            theta_j = 2 * np.pi
                    else:
                        alpha = phi_kj / phi_ij
                        if theta_k == 0 and theta_j > np.pi:
                            theta_k = 2 * np.pi
                    theta_i = np.fmod((1 - alpha) * theta_j + alpha * theta_k, 2 * np.pi)

    return u_ijk, theta_i


def compute_distance_and_angle(vertex_i, vertex_j, u, theta, triangle_cache, object_mesh, use_c, rotation_axis):
    """Euclidean update procedure for geodesic distance approximation.

    See Section 4 in:

    > [Geodesic polar coordinates on polygonal
    meshes](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2012.03187.x)
    > Melvær, Eivind Lyche, and Martin Reimers.

    **Input**

    - The vertex for which we want to update the distance: `vertex_i`.
    - Another `vertex_j` in the triangle of `vertex_i`, s.t. we have an edge `(vertex_i, vertex_j)`.
    - The currently known distances `u`.
    - The currently known angular coordinates `theta`.
    - A cache storing (or not storing) the triangles for edge `(vertex_i, vertex_j)`.
    - A loaded object mesh.
    - A flag whether to use the c-extension.

    **Output**

    - The updated distance to `vertex_i`.

    """

    if triangle_cache is None:
        triangle_cache = dict()

    # Caching considered triangles to save time
    if vertex_i + vertex_j in triangle_cache.keys():
        considered_triangles = triangle_cache[vertex_i + vertex_j]
    else:
        sorted_vectors = [vertex_i, vertex_j]
        sorted_vectors.sort()
        edge_indices = np.where(np.logical_and(
            object_mesh.edges_sorted[:, 0] == sorted_vectors[0],
            object_mesh.edges_sorted[:, 1] == sorted_vectors[1]
        ))[0]
        face_indices = object_mesh.edges_face[edge_indices]
        considered_triangles = object_mesh.faces[face_indices]
        triangle_cache[vertex_i + vertex_j] = considered_triangles

    updates = []
    for triangle in considered_triangles:
        vertex_k = [v for v in triangle if v not in [vertex_i, vertex_j]][0]
        # We need to know the distance to `vertex_k`
        if u[vertex_k] < np.inf and theta[vertex_k] >= 0.:
            u_ijk, phi_i = compute_u_ijk_and_angle(
                vertex_i, vertex_j, vertex_k, u, theta, object_mesh, use_c, rotation_axis
            )
            updates.append((u_ijk, phi_i))
    if not updates:
        return np.inf, -1.0, triangle_cache
    else:
        u_ijk, phi_i = min(updates)
        return u_ijk, phi_i, triangle_cache


def compute_vector_angle(vector_a, vector_b, rotation_axis):
    """Compute the angle between two vectors.

    **Input**

    - The first vector
    - The second vector

    **Output**

    - The angle between `vector_a` and `vector_b`

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


def get_neighbors(vertex, object_mesh, graph=None):
    """Calculates the one-hop neighbors of a vertex.

    See: https://github.com/mikedh/trimesh/issues/52

    **Input**

    - The index of the vertex for which the neighbor indices shall be computed.
    - A loaded object mesh.
    - The graph representation of the given object mesh.

    **Output**

    - A list of neighboring vertex-indices.

    """

    if graph:
        return list(graph[vertex].keys()), graph
    else:
        mesh_edges = object_mesh.edges_unique
        graph = nx.from_edgelist(mesh_edges)
        return list(graph[vertex].keys()), graph


def initialize_neighborhood(source_point, u, theta, object_mesh, graph, use_c):
    """Compute the initial radial and angular coordinates around a source point.

    Angle coordinates are always given w.r.t. some reference direction. The choice of a reference
    direction can be arbitrary. Here, we choose the vector `x - source_point` with `x` being the
    first neighbor return by `get_neighbors` as the reference direction.

    **Input**

    - The index of the source point around which a window (GPC-system) shall be established.
    - An array `u` of radial coordinates from the source point to other points in the object mesh.
    - An array `theta` of angular coordinates of neighbors from `source_point` in its window.
    - A loaded object mesh.
    - The graph representation of the loaded object mesh.
    - A flag whether to use the c-extension.

    **Output**

    - The `u` array with updated values at the neighboring nodes of `source_point`.
    - The `theta` array with updated values at the neighboring nodes of `source_point`.
    - The neighbors of `source_point`.

    """

    source_point_neighbors, graph = get_neighbors(source_point, object_mesh, graph)
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

    return u, theta, source_point_neighbors, graph, rotation_axis


def local_gpc(source_point, u_max, object_mesh, use_c, eps=0.000001, triangle_cache=None, graph=None):
    """Computes local GPC for one given source point.

    **Input**

    - The index of the source point around which a window (GPC-system) shall be established.
    - The maximal distance (e.g. radius of the patch) which a vertex may have to `source_point`.
    - A loaded object mesh.
    - A flag whether to use the c-extension.
    - An epsilon.
    - A cache storing the faces of a given edge.
    - A graph consisting of the edges of the mesh.

    **Output**

    - An array `u` of radial coordinates from the source point to other points in the object mesh.
    - An array `theta` of angular coordinates of neighbors from `source_point` in its window.

    """

    u = np.full((object_mesh.vertices.shape[0],), np.inf)
    theta = np.full((object_mesh.vertices.shape[0],), -1.0)
    u, theta, source_point_neighbors, graph, rotation_axis = initialize_neighborhood(
        source_point, u, theta, object_mesh, graph, use_c
    )
    u[source_point] = .0

    candidates = []  # heap containing vertices sorted by distance u[i]
    for neighbor in source_point_neighbors:
        candidates.append((u[neighbor], neighbor))
    heapq.heapify(candidates)

    while candidates:
        # as we work with a min-heap the shortest distance is stored in root
        j_dist, j = heapq.heappop(candidates)
        j_neighbors, _ = get_neighbors(j, object_mesh, graph)
        j_neighbors = [j for j in j_neighbors if j != source_point]

        for i in j_neighbors:
            # during computation of distance to `i` consider both triangles which contain (i, j)
            # [compare section 4.1 and algorithm 1]
            new_u_i, new_theta_i, triangle_cache = compute_distance_and_angle(
                i, j, u, theta, triangle_cache, object_mesh, use_c, rotation_axis
            )

            if new_u_i < np.inf and u[i] / new_u_i > 1 + eps:
                # also adding new_u_i which are > u_max, however
                # their neighbors are not further explored
                u[i] = new_u_i
                theta[i] = new_theta_i
                if new_u_i < u_max:
                    heapq.heappush(candidates, (new_u_i, i))

    return u, theta, triangle_cache, graph


def discrete_gpc(object_mesh, u_max=.04, eps=0.000001, use_c=False):
    """Computes approximated geodesic polar coordinates for all vertices within an object mesh.

    > [Geodesic polar coordinates on polygonal
    meshes](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2012.03187.x)
    > Melvær, Eivind Lyche, and Martin Reimers.

    **Input**

    - A loaded object mesh.

    **Output**

    - Array A with dimensions `(2, n, n)` with `n = object_mesh.vertices.shape[0]`. A[i][j][0] stores the radial
      distance (with max value `u_max`) from node `j` to origin `i` of the local GPC-system. A[i][j][1] contains
      the radial coordinate of node `j` in the local GPC-system of node `i` w.r.t. a reference direction (see
      `initialize_neighborhood` for how the reference direction is selected).

    """
    sys.stderr.write("Calculating local gpc-systems..")
    u, theta, triangle_cache, graph = [], [], dict(), None
    for vertex_idx in tqdm.tqdm(range(object_mesh.vertices.shape[0])):
        u_v, theta_v, triangle_cache, graph = local_gpc(
            vertex_idx, u_max, object_mesh, use_c, eps, triangle_cache, graph
        )
        u.append(u_v)
        theta.append(theta_v)

    return np.stack([u, theta], axis=-1)
