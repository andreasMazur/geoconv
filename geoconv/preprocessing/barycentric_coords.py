import numpy as np
import scipy as sp
import sys
import warnings


def polar_to_cart(angle, scale=1.):
    """Returns x and y for a given angle.

    Parameters
    ----------
    angle: float
        The angular coordinate
    scale: float
        The radial coordinate

    Returns
    -------
    (float, float):
        The x-coordinate and y-coordinate

    """
    return scale * np.cos(angle), scale * np.sin(angle)


def get_points_from_polygons(polygons):
    """Returns all corner-points of the given polygons.

    Parameters
    ----------
    polygons: np.ndarray
        A 3D-array Arr[n, m, d], containing 'n' polygons with 'm' points of dimensionality 'd'.

    Returns
    -------
    np.ndarray:
        A 2D-array Arr[x, d], containing 'x' unique points with dimensionality 'd'.
    """

    return np.unique(polygons.reshape((-1, polygons.shape[2])), axis=0)


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


def find_triangle(vertex, gpc_triangles, gpc_faces):
    """Finds the triangle in which the given vertex fell into and returns the b.-coordinates

    Parameters
    ----------
    vertex: np.ndarray
        The vertex for which the triangle is searched for in cartesian coordinates.
    gpc_triangles: np.ndarray
        The triangles of the GPC-system given in cartesian coordinates.
    gpc_faces: np.ndarray
        The associated indices for the vertices in 'gpc_triangles'

    Returns
    -------
    (np.ndarray, np.ndarray)
        The first returned array contains the barycentric coordinates. The second returned array contains the indices
        of the vertices to which the barycentric coordinates belong. In case of 'vertex' not falling into any triangle,
        two zero arrays are returned. I.e. there will be no feature taken into consideration for this kernel vertex.
    """

    all_gpc_nodes = get_points_from_polygons(gpc_triangles)
    kd_tree = sp.spatial.KDTree(all_gpc_nodes)

    for k in range(1, all_gpc_nodes.shape[0]):
        # Get triangles of nearest neighbor
        if k == 1:
            nearest_neighbor_idx = kd_tree.query(vertex, k=k)[1]
        else:
            nearest_neighbor_idx = kd_tree.query(vertex, k=k)[1][k - 1]
        nearest_neighbor = all_gpc_nodes[nearest_neighbor_idx]

        tri_indices = np.unique(np.where(gpc_triangles == nearest_neighbor)[0])
        nearest_neighbor_triangles = gpc_triangles[tri_indices]
        nearest_neighbor_faces = gpc_faces[tri_indices]

        # Check for each triangle whether the query vertex fell into it
        for tri_idx in range(nearest_neighbor_triangles.shape[0]):
            b_coordinates, lies_within = compute_barycentric(vertex, nearest_neighbor_triangles[tri_idx])
            if lies_within:
                return b_coordinates, nearest_neighbor_faces[tri_idx]

    # If vertex falls in no triangle: No feature vector is given
    return np.array([0., 0., 0.]), np.array([0, 0, 0])


def barycentric_coordinates_kernel(kernel, gpc_triangles, gpc_faces):
    """Computes the barycentric coordinates for all kernel vertices.

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

    # Find Barycentric coordinates iteratively for every kernel vertex
    n_radial = kernel.shape[0]
    n_angular = kernel.shape[1]
    b_coordinates = np.zeros((n_radial, n_angular, 3, 2))

    # Convert coordinates from polar to cartesian
    for tri_idx in range(gpc_triangles.shape[0]):
        for point_idx in range(gpc_triangles.shape[1]):
            rho, theta = gpc_triangles[tri_idx, point_idx]
            gpc_triangles[tri_idx, point_idx] = polar_to_cart(theta, scale=rho)

    for i in range(n_radial):
        for j in range(n_angular):
            bc, face = find_triangle(kernel[i, j], gpc_triangles, gpc_faces)
            b_coordinates[i, j, :, 0] = face
            b_coordinates[i, j, :, 1] = bc

    return b_coordinates


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
