import numpy as np
import scipy
import warnings

from GeodesicPolarMap.discrete_gpc import local_gpc

NUMPY_ROUNDING_DECIMAL = 10


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

    return np.round(B, decimals=NUMPY_ROUNDING_DECIMAL)


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
    return np.round(coordinates, decimals=NUMPY_ROUNDING_DECIMAL)


def compute_barycentric(query_vertex, triangle):
    """Computes barycentric coordinates.

    Compare: https://blackpawn.com/texts/pointinpoly/
    """

    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = query_vertex - triangle[0]

    dot00, dot01, dot02 = v0.dot(v0), v0.dot(v1), v0.dot(v2)
    dot11, dot12 = v1.dot(v1), v1.dot(v2)

    x = 1 / (dot00 * dot11 - dot01 * dot01)
    point_2_weight = (dot11 * dot02 - dot01 * dot12) * x
    point_1_weight = (dot00 * dot12 - dot01 * dot02) * x
    point_0_weight = 1 - point_2_weight - point_1_weight

    is_inside_triangle = point_2_weight >= 0 and point_1_weight >= 0 and point_2_weight + point_1_weight < 1

    return (point_0_weight, point_1_weight, point_2_weight), is_inside_triangle


def compute_barycentric_help(query_vertex, triangles, local_gpc_system):

    result = None
    held_back = []
    for triangle in triangles:
        # Determine 2D geodesic coordinates of the considered triangle
        geodesic_coordinates = local_gpc_system[triangle]
        if not np.any(geodesic_coordinates == np.inf):
            # Compute the barycentric coordinates of the triangle
            (b0, b1, b2), query_inside_tri = compute_barycentric(query_vertex, geodesic_coordinates)
            if query_inside_tri:
                result = [b0, triangle[0]], [b1, triangle[1]], [b2, triangle[2]]
        else:
            held_back.append(triangle)

    return result, held_back

def compute_barycentric_triangles(query_vertex, triangles, local_gpc_system, source_point, object_mesh, gpc_radius):
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
    result, held_back = compute_barycentric_help(query_vertex, triangles, local_gpc_system)
    if result is not None:
        return result

    # TODO: Recompute local GPC-system with larger radius and check
    # TODO: whether triangles are now fully included
    # raise RuntimeError(
    #     f"Your local GPCs do not include all vertices necessary to capture required triangles. Consider to shrink your"
    #     f" kernel radius or to increase your local GPC-system's radius."
    # )
    triangle_cache, graph = dict(), None
    iteration = 0
    while result is None:
        # Compute larger GPC-system
        gpc_radius *= 2
        print(f"Iteration: {iteration} with radius: {gpc_radius}")
        u, theta, triangle_cache, graph = local_gpc(
            source_point, gpc_radius, object_mesh, triangle_cache=triangle_cache, graph=graph
        )
        local_gpc_system = np.round(np.stack([u, theta], axis=-1), decimals=NUMPY_ROUNDING_DECIMAL)
        local_gpc_system = polar_2_cartesian(np.expand_dims(local_gpc_system, axis=0))[0]

        result, held_back = compute_barycentric_help(query_vertex, triangles, local_gpc_system)

    print("Done.\n\n")

    return result


def barycentric_weights_local_gpc(local_gpc_system, kernel, object_mesh, source_point):
    """Computes barycentric weights for a kernel placed in a source point.

    **Input**

    -

    **Output**

    -

    """
    # Consider only the vertices which we have coordinates for
    v_with_coords = local_gpc_system != np.inf
    v_with_coords = local_gpc_system[np.logical_and(v_with_coords[:, 0], v_with_coords[:, 1])]
    kd_tree = scipy.spatial.KDTree(v_with_coords)

    kernel = kernel.reshape((kernel.shape[0] * kernel.shape[1], 2))
    barycentric_coordinates = []
    for k in kernel:
        # Query the KD-tree for the vertex index of the nearest neighbor of `k`
        _, gpc_system_idx = kd_tree.query(k)
        nearest_neighbor = v_with_coords[gpc_system_idx]
        row_indices, _ = np.where(local_gpc_system == nearest_neighbor)
        queried_vertex = row_indices[0]

        # Query for the triangle indices of all triangles that contain `queried_vertex`
        triangle_indices = [x for x in object_mesh.vertex_faces[queried_vertex] if x != -1]
        triangles = object_mesh.faces[triangle_indices]
        b_coords = compute_barycentric_triangles(k, triangles, local_gpc_system, source_point, object_mesh, gpc_radius=.04)
        barycentric_coordinates.append(b_coords)

    return barycentric_coordinates


def barycentric_weights(local_gpc_systems, kernel, object_mesh):
    """Computes barycentric weights for a kernel placed in all source points of an object mesh.

    In order to compute the barycentric weights for kernel vertices we do the following:

        1.) Translate coordinates for kernel vertices and local GPC-systems into cartesian coordinates.
        2.) For each local GPC-system:
            2.1) Compute a KD-tree.
            2.2) For each kernel vertex `k`:
                2.2.1) Find the nearest neighbor within the local GPC-system querying within the KD-tree.
                       Let this vertex be `x`.
                2.2.2) For each triangle of `x` compute the barycentric weights w.r.t. `k`.
                2.2.3) Given the barycentric weights, determine the triangle `T` that contains `k`.
                2.2.4) Store the barycentric weights of `T` w.r.t. `k`.
        3.) Store the barycentric weights in an array `W` in the following manner (compare paper below, section 4.3):
            - `W` has size `(N_v, N_rho, N_theta, N_v)`
                * `N_v = object_mesh.vertices.shape[0]` (amount vertices)
                * `N_rho = kernel.shape[0]` (amount radial coordinates)
                * `N_theta = kernel.shape[1]` (amount angular coordinates)

    > [Multi-directional geodesic neural networks via equivariant
    convolution](https://dl.acm.org/doi/abs/10.1145/3272127.3275102)
    > Adrien Poulenard and Maks Ovsjanikov.

    **Input**

    - Array that contains the local GPC-systems for every considered node in an object mesh (compare output of
      `GeodesicPolarMap.discrete_gpc`)
    - Array that contains the polar coordinates of the kernel's vertices (compare output of `create_kernel_matrix`)

    **Output**

    -

    """

    local_gpc_systems = np.round(local_gpc_systems, decimals=NUMPY_ROUNDING_DECIMAL)
    local_gpc_systems = polar_2_cartesian(local_gpc_systems)
    kernel = polar_2_cartesian(kernel)

    for source_point, gpc_system in enumerate(local_gpc_systems):
        b_coords = barycentric_weights_local_gpc(gpc_system, kernel, object_mesh, source_point)
        print()
