import numpy as np
import scipy as sp
import sys
import warnings


def create_kernel_matrix(n_radial, n_angular, radius):
    """Creates a kernel matrix with radius `radius` and `n_radial` radial- and `n_angular` angular coordinates.

    Parameters
    ----------
    n_radial: int
        The amount of radial coordinates
    n_angular: int
        The amount of angular coordinates
    radius: float
        The radius of the kernel

    Returns
    -------
    np.ndarray
        A kernel matrix K with K[i, j] containing polar coordinates (radial, angular) of point (i. j)
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
    """Computes barycentric coordinates

    Compare: https://blackpawn.com/texts/pointinpoly/

    Parameters
    ----------
    query_vertex: np.ndarray
        1D-array that contains query-vertex in polar coordinates
    triangle: np.ndarray
        2D-array that depicts a triangle in polar coordinates

    Returns
    -------
    ((np.float32, np.float32, np.float32), bool)
        A tuple containing a triple and a boolean. The triple contains the barycentric coordinates for the vertices of
        the triangle. The boolean tells whether the query-vertex is within the triangle.
    """

    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = query_vertex - triangle[0]

    dot00, dot01, dot02 = v0.dot(v0), v0.dot(v1), v0.dot(v2)
    dot11, dot12 = v1.dot(v1), v1.dot(v2)

    denominator = dot00 * dot11 - dot01 * dot01
    if denominator > 0:
        point_2_weight = (dot11 * dot02 - dot01 * dot12) / denominator
        point_1_weight = (dot00 * dot12 - dot01 * dot02) / denominator
        point_0_weight = 1 - point_2_weight - point_1_weight

        is_inside_triangle = point_2_weight >= 0 and point_1_weight >= 0 and point_2_weight + point_1_weight <= 1

        return (point_0_weight, point_1_weight, point_2_weight), is_inside_triangle
    else:
        return (None, None, None), False


def polar_to_cartesian(coordinate_array):
    """Compute cartesian coordinates for given polar coordinates

    Parameters
    ----------
    coordinate_array: np.ndarray
        A 2D-array with coordinate_array[:, 0] containing radial coordinates and coordinate_array[:, 1] containing
        angular coordinates

    Returns
    -------
    np.ndarray
        A 2D-array with cartesian[:, 0] containing x-coordinates and cartesian[:, 1] containing y-coordinates
    """
    cartesian = np.zeros_like(coordinate_array)
    cartesian[:, 0] = coordinate_array[:, 0] * np.cos(coordinate_array[:, 1])
    with warnings.catch_warnings():
        # If this function is applied onto the result of `discrete_gpc` the array `coordinate_array[:, :, 0]` may
        # contain `np.inf`-values. Multiplying them results in a warning which is unnecessary here.
        warnings.filterwarnings("ignore", "invalid value encountered in multiply")
        cartesian[:, 1] = coordinate_array[:, 0] * np.sin(coordinate_array[:, 1])

    return cartesian


def determine_gpc_triangles(object_mesh, local_gpc_system):
    """Get triangles and faces which are contained within a given local GPC-system

    Parameters
    ----------
    object_mesh: trimesh.Trimesh
        The object mesh
    local_gpc_system: np.ndarray
        2D-array containing the geodesic polar coordinates for each vertex in the local GPC-system

    Returns
    -------
    (np.ndarray, np.ndarray):
        Two arrays. The first 3D-array contains all triangles that are entirely contained in the GPC-system. The second
        2D-array contains the same triangles described in node-indices.
    """

    # Filter triangles such that only those remain that are entirely described by local GPC-system
    local_triangles = local_gpc_system[object_mesh.faces]
    valid_triangle_indices = list(set(range(local_triangles.shape[0])) - set(np.where(local_triangles == np.inf)[0]))
    valid_gpc_triangles = local_triangles[valid_triangle_indices]
    valid_gpc_faces = np.array(object_mesh.faces[valid_triangle_indices])

    return valid_gpc_triangles, valid_gpc_faces


def barycentric_coordinates_kernel(kernel, gpc_triangles, gpc_faces):
    """Find suiting Barycentric coordinates for kernel-vertices among all triangles of a GPC-system

    In order to compute the barycentric coordinates for kernel vertices we do the following:

    1.) For each local GPC-system:
        1.1) Compute a KD-tree.
        1.2) For each kernel vertex `k`:
            1.2.1) Find the nearest neighbor within the local GPC-system querying within the KD-tree. Let this vertex be
                   `x`.
            1.2.2) For each triangle of `x` compute the barycentric coordinates w.r.t. `k`.
            1.2.3) Given the barycentric coordinates, determine the triangle `T` that contains `k`. IF there is not
                   such `T` then go back to (1.2.1) and compute the next nearest neighbor apart from `x` and repeat.
            1.2.4) Store the barycentric coordinates of `T` w.r.t. `k`.
    2.) Store the barycentric coordinates.

    Parameters
    ----------
    kernel: np.ndarray
        A 3D-array containing the kernel-vertices described in cartesian coordinates
        - kernel[i, j] contains cartesian kernel coordinates for kernel vertex referenced by i-th radial and j-th
          angular coordinate
    gpc_triangles: np.ndarray
        A 3D-array containing the triangles of the considered GPC-system
        - gpc_triangles[i] contains i-th triangle depicted in cartesian coordinates
    gpc_faces: np.ndarray
        A 2D-array containing the triangles of the considered GPC-system
        - gpc_faces[i] contains i-th triangle depicted in vertex indices

    Returns
    -------
    np.ndarray
        A 4D-array barycentric
        - barycentric[i, j, :3, 0] contains vertex indices that have Barycentric coordinates for kernel-vertex (i, j)
        - barycentric[i, j, :3, 1] contains respective Barycentric coordinates of vertices barycentric[i, j, :3, 0] for
          kernel-vertex (i, j)
        Note that the radial- and angular dimension have been switched! This allows the geodesic convolution to benefit
        in efficiency from tensorflow broadcasting.
    """
    # Find closest point to query vertex in considered GPC-system
    all_gpc_nodes = np.unique(gpc_triangles.reshape((-1, 2)), axis=0)
    gpc_kd_tree = sp.spatial.KDTree(all_gpc_nodes)

    # Find Barycentric coordinates iteratively for every kernel vertex
    n_radial = kernel.shape[0]
    n_angular = kernel.shape[1]
    barycentric = np.zeros((n_radial, n_angular, 3, 2))
    for i in range(n_radial):
        for j in range(n_angular):
            query_vertex = kernel[i, j]
            nth_closest_vertex = 1
            is_within = False

            while not is_within:
                # Find n-th closest node among GPC-system vertices
                _, closest_node_idx = gpc_kd_tree.query(query_vertex, k=nth_closest_vertex)
                if nth_closest_vertex > 1:
                    closest_node_idx = closest_node_idx[nth_closest_vertex - 1]
                closest_node = all_gpc_nodes[closest_node_idx]

                # Find triangles of the closest node and compute Barycentric coordinates for them
                considered_triangle_indices = np.unique(np.where(gpc_triangles == closest_node)[0])
                for triangle_idx in considered_triangle_indices:
                    face = gpc_faces[triangle_idx]
                    barycentric_coords, is_within = compute_barycentric(query_vertex, gpc_triangles[triangle_idx])
                    if is_within:
                        # Store Barycentric coordinates and respective node indices
                        for idx in range(3):
                            barycentric[i, j, idx, 0] = face[idx]
                            barycentric[i, j, idx, 1] = barycentric_coords[idx]
                        break

                if not is_within:
                    # Look for triangles of the next closest point
                    nth_closest_vertex += 1

                # Fallback: If there is no triangle containing the kernel vertex then use the closest vertex as
                #           approximation
                if nth_closest_vertex >= all_gpc_nodes.shape[0] and not is_within:
                    # Get closest node
                    _, closest_node_idx = gpc_kd_tree.query(query_vertex)
                    closest_node = all_gpc_nodes[closest_node_idx]

                    # Get id of closest node
                    node_id = np.argwhere(gpc_triangles == closest_node)[0]
                    node_id = gpc_faces[node_id[0], node_id[1]]

                    # Give closest node weight 1. (interpolation value equals signal at that vertex)
                    barycentric[i, j, 0, 0] = node_id
                    barycentric[i, j, 0, 1] = 1.
                    break

    return barycentric


def barycentric_coordinates(object_mesh, gpc_systems, n_radial=2, n_angular=4, radius=0.05, verbose=True):
    """Compute the barycentric coordinates for the given GPC-systems

    Parameters
    ----------
    object_mesh: trimesh.Trimesh
        The corresponding object mesh for the GPC-systems
    gpc_systems: np.ndarray
        3D-array in which the i-th entry corresponds to the GPC-system centered in vertex i from the given object mesh
        - contains polar coordinates (radial, angular)
        - just use the output of layers.preprocessing.discrete_gpc.discrete_gpc
    n_radial: int
        The amount of radial coordinates of the kernel you wish to use
    n_angular: int
        The amount of angular coordinates of the kernel you wish to use
    radius: float
        The radius of the kernel of the kernel you wish to use
    verbose: bool
        Whether to print progress on terminal

    Returns
    -------
    A 5D-array containing the Barycentric coordinates for each kernel vertex and each GPC-system. It has the following
    structure:
        B[a, b, c, d, e]:
            - a: References GPC-system centered in vertex `a` of object mesh `object_mesh`
            - b: References the b-th radial coordinate of the kernel
            - c: References the c-th angular coordinate of the kernel
            - B[a, b, c, :, 0]: Returns the **indices** of the nodes that construct the triangle containing the kernel
                                vertex (b, c) in GPC-system centered in node `a`
            - B[a, b, c, :, 1]: Returns the **Barycentric coordinates** of the nodes that construct the triangle
                                containing the kernel vertex (b, c) in GPC-system centered in node `a`
    """

    # Define kernel vertices at which interpolation values will be needed
    kernel = create_kernel_matrix(n_radial=n_radial, n_angular=n_angular, radius=radius)

    # We lay the kernel once onto every local GPC-system centered in origin
    amt_gpc_systems = gpc_systems.shape[0]
    all_barycentric_coords = np.zeros((amt_gpc_systems, n_radial, n_angular, 3, 2))

    # Translate all coordinates into cartesian coordinates such that we can work with KD-trees
    kernel = polar_to_cartesian(kernel.reshape((-1, 2))).reshape((n_radial, n_angular, 2))
    for gpc_system_idx, gpc_system in enumerate(gpc_systems):
        if verbose:
            sys.stdout.write(
                f"\rCurrently computing Barycentric coordinates for GPC-system centered in vertex {gpc_system_idx}"
            )
        gpc_system = polar_to_cartesian(gpc_system)

        # Determine triangles which are contained within the currently considered local GPC-system
        contained_gpc_triangles, contained_gpc_faces = determine_gpc_triangles(object_mesh, gpc_system)

        # Store Barycentric coordinates for kernel in i-th GPC-system
        barycentric_coords = barycentric_coordinates_kernel(kernel, contained_gpc_triangles, contained_gpc_faces)
        all_barycentric_coords[gpc_system_idx] = barycentric_coords

    return all_barycentric_coords.astype(np.float32)
