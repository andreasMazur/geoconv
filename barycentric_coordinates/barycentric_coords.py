import numpy as np
import scipy
import warnings
import tqdm


def polar_2_cartesian(coordinate_array):
    """Compute cartesian coordinate for given polar coordinates.

    **Input**

    - Array A of size (n_radial, n_angular, 2) with A[i, j] containing (radial_coordinate, angular_coordinate).

    **Output**

    - Array B of same size as input array. However, polar coordinates are replaced with cartesian coordinates. That is,
      B[i, j] contains cartesian coordinates (x, y) for polar coordinates A[i, j].

    """
    B = np.zeros_like(coordinate_array)
    B[:, :, 0] = coordinate_array[:, :, 0] * np.cos(coordinate_array[:, :, 1])
    with warnings.catch_warnings():
        # If this function is applied onto the result of `discrete_gpc` the array `coordinate_array[:, :, 0]` may
        # contain `np.inf`-values. Multiplying them results in a warning which is unnecessary here.
        warnings.filterwarnings("ignore", "invalid value encountered in multiply")
        B[:, :, 1] = coordinate_array[:, :, 0] * np.sin(coordinate_array[:, :, 1])

    return np.round(B, decimals=10)


def create_kernel_matrix(n_radial, n_angular, radius):
    """Creates a kernel matrix with radius `radius` and `n_radial` radial- and `n_angular` angular coordinates.

    **Input**

    - The amount of radial coordinates.
    - The amount of angular coordinates.
    - The radius of the kernel.

    **Output**

    - A kernel matrix K with K[i, j] containing polar coordinates (radial, angular) of point (i. j).

    """

    coordinates = np.zeros((n_radial, n_angular, 2))
    for j in range(1, n_radial + 1):
        radial_coordinate = (j * radius) / n_radial
        for k in range(1, n_angular + 1):
            angular_coordinate = (2 * k * np.pi) / n_angular
            coordinates[j - 1, k - 1, 0] = radial_coordinate
            coordinates[j - 1, k - 1, 1] = angular_coordinate
    return coordinates


def compute_barycentric(query_vertex, triangle):
    """Computes barycentric coordinates.

    Compare: https://blackpawn.com/texts/pointinpoly/
    """

    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = query_vertex - triangle[0]

    dot00, dot01, dot02 = v0.dot(v0), v0.dot(v1), v0.dot(v2)
    dot11, dot12 = v1.dot(v1), v1.dot(v2)

    denominator = dot00 * dot11 - dot01 * dot01
    if denominator > 0:
        x = 1 / denominator
        point_2_weight = (dot11 * dot02 - dot01 * dot12) * x
        point_1_weight = (dot00 * dot12 - dot01 * dot02) * x
        point_0_weight = 1 - point_2_weight - point_1_weight

        is_inside_triangle = point_2_weight >= 0 and point_1_weight >= 0 and point_2_weight + point_1_weight < 1

        return (point_0_weight, point_1_weight, point_2_weight), is_inside_triangle
    else:
        return (None, None, None), False


def compute_barycentric_triangles(kernel_vertex, faces, local_gpc_system):
    """Looks for the triangle which contains the query vertex and computes the corresponding barycentric coordinates.

    **Input**

    - The query vertex.
    - Triangles, which may include the query vertex. Usually the triangles of the closest point to the query vertex.
    - The local GPC-system in cartesian coordinates within which the barycentric coordinates are computed.

    **Output**

    - The barycentric coordinates for the query vertex combined with their corresponding vertex index.

    **Raises**

    - A runtime error if the query index is contained within a triangle that is not captured entirely by the
      local GPC-system.

    """

    result = None
    held_back = []
    for face in faces:
        # Determine 2D geodesic coordinates of the considered triangle
        geodesic_coordinates = local_gpc_system[face]
        if not np.any(geodesic_coordinates == np.inf):
            # Compute the barycentric coordinates of the triangle
            (b0, b1, b2), query_inside_tri = compute_barycentric(kernel_vertex, geodesic_coordinates)
            if query_inside_tri:
                result = (b0, face[0], b1, face[1], b2, face[2])
        else:
            held_back.append(face)

    return result


def barycentric_coords_local_gpc(local_gpc_system, kernel, om_faces, om_vertex_faces):
    """Computes barycentric coordinates for a kernel placed in a source point.

    **Input**

    - The local GPC system translated into cartesian coordinates (required for KD-tree).
    - The kernel coordinates translated into cartesian coordinates (required for KD-tree).
    - The considered object mesh

    **Output**

    - A tuple `t` containing the following elements:
        * `t[0]`: Kernel index `i` referring to the radial coordinate
        * `t[1]`: Kernel index `j` referring to the angular coordinate
        * `t[2]`: 1. Barycentric coordinate
        * `t[3]`: Corresponding node index for 1. Barycentric coordinate
        * `t[4]`: 2. Barycentric coordinate
        * `t[5]`: Corresponding node index for 2. Barycentric coordinate
        * `t[6]`: 3. Barycentric coordinate
        * `t[7]`: Corresponding node index for 3. Barycentric coordinate

    If a kernel vertex does not fall into any triangle in the local GPC-system, then the closest neighbor of
    gets a `1.0`-barycentric coordinate assigned.

    """
    kernel = np.round(kernel, decimals=10)

    # Trimesh object meshes cache queries. This takes time. Normal numpy arrays are faster here.

    # Consider only the vertices which we have coordinates for
    v_with_coords = local_gpc_system != np.inf
    v_with_coords = local_gpc_system[np.logical_and(v_with_coords[:, 0], v_with_coords[:, 1])]
    kd_tree = scipy.spatial.KDTree(v_with_coords)

    barycentric_coordinates = []
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            k = kernel[i, j]
            b_coords = None
            try_ = 1
            while not b_coords:
                # Query the KD-tree for the vertex index of the nearest neighbor of `k`
                _, nn_idx = kd_tree.query(k, k=try_)
                if try_ > 1:
                    nn_idx = nn_idx[try_ - 1]
                nearest_neighbor = v_with_coords[nn_idx]
                row_indices, _ = np.where(local_gpc_system == nearest_neighbor)

                # Check for validity of the GPC
                if row_indices.shape != (2,):
                    zipped_rows = np.stack([row_indices, np.append(row_indices[1:], row_indices[0])], axis=-1)[:-1]
                    matches = np.where(zipped_rows[:, 0] == zipped_rows[:, 1])[0]
                    if matches.shape[0] > 1:
                        raise RuntimeError("Multiple occurrences of the same coordinate in a GPC!")
                    else:
                        queried_vertex = row_indices[matches[0]]
                else:
                    queried_vertex = row_indices[0]

                # Query for the triangle indices of all triangles that contain `queried_vertex`
                face_indices = om_vertex_faces[queried_vertex]
                face_indices = face_indices[face_indices != -1]
                faces = om_faces[face_indices]
                b_coords = compute_barycentric_triangles(k, faces, local_gpc_system)
                if not b_coords:
                    try_ += 1
                    if try_ > v_with_coords.shape[0]:
                        # Failsafe: If no triangle was found, then use the closest vertex as approximation.
                        _, nn_idx = kd_tree.query(k)
                        b_coords = (1.0, nn_idx, None, None, None, None)

            barycentric_coordinates.append((i, j) + b_coords)

    return barycentric_coordinates


def barycentric_coordinates(local_gpc_systems, kernel, object_mesh):
    """Computes barycentric coordinates for a kernel placed in all source points of an object mesh.

    In order to compute the barycentric coordinates for kernel vertices we do the following:

        1.) Translate coordinates for kernel vertices and local GPC-systems into 2D cartesian coordinates.
        2.) For each local GPC-system:
            2.1) Compute a KD-tree.
            2.2) For each kernel vertex `k`:
                2.2.1) Find the nearest neighbor within the local GPC-system querying within the KD-tree.
                       Let this vertex be `x`.
                2.2.2) For each triangle of `x` compute the barycentric coordinates w.r.t. `k`.
                2.2.3) Given the barycentric coordinates, determine the triangle `T` that contains `k`. IF there is not such
                       `T` then go back to (2.2.1) and compute the next nearest neighbor apart from `x` and repeat.
                2.2.4) Store the barycentric coordinates of `T` w.r.t. `k`.
        3.) Store the barycentric coordinates in an array `E` in the following manner (compare paper below, section 4.3):
            - `E` has size `(N_v, N_rho, N_theta, N_v)`
                * `N_v = object_mesh.vertices.shape[0]` (amount vertices)
                * `N_rho = kernel.shape[0]` (amount radial coordinates)
                * `N_theta = kernel.shape[1]` (amount angular coordinates)

    > [Multi-directional geodesic neural networks via equivariant
    convolution](https://dl.acm.org/doi/abs/10.1145/3272127.3275102)
    > Adrien Poulenard and Maks Ovsjanikov.

    **Input**

    - Array that contains the local GPC-systems for every considered node in an object mesh (compare output of
      `geodesic_polar_coordinates.discrete_gpc`)
    - Array that contains the polar coordinates of the kernel's vertices (compare output of `create_kernel_matrix`)

    **Output**

    - 3-dimensional array `E` with size `(#gpc_systems, #radial_coord's * #angular_coord's, 8)`. Per GPC-system it
      stores for every kernel-vertex the tuple given by `barycentric_coords_local_gpc`. These tuples described the
      Barycentric coordinates for each kernel-vertex and the corresponding node indices.

    """
    local_gpc_systems = polar_2_cartesian(local_gpc_systems)
    kernel = polar_2_cartesian(kernel)

    faces = np.copy(object_mesh.faces)
    vertex_faces = np.copy(object_mesh.vertex_faces)

    # E = np.zeros((amt_nodes, amt_radial_coordinates, amt_angular_bins, amt_nodes))
    E = []
    for gpc_system in tqdm.tqdm(local_gpc_systems):
        E.append(barycentric_coords_local_gpc(gpc_system, kernel, faces, vertex_faces))

    return np.array(E)
