from tqdm import tqdm

import numpy as np


def determine_gpc_triangles(object_mesh, gpc_system):
    """Get triangles and faces which are contained within a given local GPC-system

    Parameters
    ----------
    object_mesh: trimesh.Trimesh
        The object mesh
    gpc_system: np.ndarray
        2D-array containing the coordinates for each vertex in the local GPC-system

    Returns
    -------
    (np.ndarray, np.ndarray):
        Two arrays. The first 3D-array contains all triangles that are entirely contained in the GPC-system. The second
        2D-array contains the same triangles described in node-indices. Depending on whether the GPC-system has geodesic
        polar- or cartesian coordinates, the triangles will have geodesic polar- or cartesian coordinates.
    """
    triangles = gpc_system[object_mesh.faces]
    indices = []
    for idx, tri in enumerate(triangles):
        if not (np.inf in tri or -1. in tri):
            indices.append(idx)

    return triangles[indices], object_mesh.faces[indices]


def compute_barycentric(query_vertex, triangle):
    """Computes barycentric coordinates

    Compare: https://blackpawn.com/texts/pointinpoly/

    Parameters
    ----------
    query_vertex: np.ndarray
        1D-array that contains query-vertex in cartesian coordinates
    triangle: np.ndarray
        2D-array that depicts a triangle in cartesian coordinates

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

        is_inside_triangle = point_2_weight > 0 and point_1_weight > 0 and point_2_weight + point_1_weight <= 1

        return (point_0_weight, point_1_weight, point_2_weight), is_inside_triangle
    else:
        return (None, None, None), False


def prep_interpolation(gpc_system, object_mesh):
    """Retrieves the triangles that are contained within a given GPC-system

    Parameters
    ----------
    gpc_system:
        The current GPC-system from which the triangles are required
    object_mesh:
        The underlying object mesh

    Returns
    -------
    (np.ndarray, np.ndarray)
        The first array contains the triangles that are contained within the given GPC-system in cartesian coordinates.
        The second array contains the indices of the nodes from the triangles in the according order.
    """

    gpc_triangles, gpc_triangles_node_indices = determine_gpc_triangles(object_mesh, gpc_system)
    for tri_idx in range(gpc_triangles.shape[0]):
        for point_idx in range(gpc_triangles.shape[1]):
            rho, theta = gpc_triangles[tri_idx, point_idx]
            gpc_triangles[tri_idx, point_idx] = polar_to_cart(theta, scale=rho)

    return gpc_triangles, gpc_triangles_node_indices


def interpolation(query_point, gpc_triangles, gpc_triangles_node_indices):
    """Interpolates a query point within a GPC-system

    Parameters
    ----------
    query_point: np.ndarray
        A query point in cartesian coordinates
    gpc_triangles: np.ndarray
        The triangles returned by 'prep_interpolation'
    gpc_triangles_node_indices: np.ndarray
        The indices returned by 'prep_interpolation'

    Returns
    -------
    (np.ndarray, np.ndarray):
        The first returned array contains the barycentric coordinates. The second returned array contains the
        indices of the vertices to which the barycentric coordinates belong. In case of 'vertex' not falling into
        any triangle, two zero arrays are returned. I.e. there will be no feature taken into consideration for this
        kernel vertex.
    """
    for idx, triangle in enumerate(gpc_triangles):
        b_coordinates, lies_within = compute_barycentric(query_point, triangle)
        if lies_within:
            return b_coordinates, gpc_triangles_node_indices[idx]
    return np.array([0., 0., 0.]), np.array([0, 0, 0])


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


def create_kernel_matrix(n_radial, n_angular, radius, in_cart=False):
    """Creates a kernel matrix with radius `radius` and `n_radial` radial- and `n_angular` angular coordinates.

    Parameters
    ----------
    n_radial: int
        The amount of radial coordinates
    n_angular: int
        The amount of angular coordinates
    radius: float
        The radius of the kernel
    in_cart: bool
        If True, then the kernel matrix contains cartesian coordinates

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

    if in_cart:
        for rc in range(coordinates.shape[0]):
            for ac in range(coordinates.shape[1]):
                coordinates[rc, ac] = polar_to_cart(coordinates[rc, ac, 1], coordinates[rc, ac, 0])

    return coordinates


def compute_barycentric_coordinates(object_mesh, gpc_systems, n_radial=2, n_angular=4, radius=0.05, verbose=True):
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
    kernel_matrix = create_kernel_matrix(n_radial=n_radial, n_angular=n_angular, radius=radius, in_cart=True)

    n_gpc_systems = gpc_systems.shape[0]
    barycentric_coordinates = np.zeros((n_gpc_systems, n_radial, n_angular, 3, 2))

    if verbose:
        for gpc_system_idx in tqdm(range(n_gpc_systems), postfix=f"Computing barycentric coordinates"):
            gpc_triangles, gpc_triangles_node_indices = prep_interpolation(gpc_systems[gpc_system_idx], object_mesh)

            for radial_coordinate in range(n_radial):
                for angular_coordinate in range(n_angular):

                    bc, indices = interpolation(
                        kernel_matrix[radial_coordinate, angular_coordinate],
                        gpc_triangles,
                        gpc_triangles_node_indices
                    )

                    barycentric_coordinates[gpc_system_idx, radial_coordinate, angular_coordinate, :, 0] = indices
                    barycentric_coordinates[gpc_system_idx, radial_coordinate, angular_coordinate, :, 1] = bc
    else:
        for gpc_system_idx in range(n_gpc_systems):
            gpc_triangles, gpc_triangles_node_indices = prep_interpolation(gpc_systems[gpc_system_idx], object_mesh)

            for radial_coordinate in range(n_radial):
                for angular_coordinate in range(n_angular):
                    bc, indices = interpolation(
                        kernel_matrix[radial_coordinate, angular_coordinate],
                        gpc_triangles,
                        gpc_triangles_node_indices
                    )

                    barycentric_coordinates[gpc_system_idx, radial_coordinate, angular_coordinate, :, 0] = indices
                    barycentric_coordinates[gpc_system_idx, radial_coordinate, angular_coordinate, :, 1] = bc

    return barycentric_coordinates
