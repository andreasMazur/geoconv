import numpy as np
import sys


def polar_to_cart(angles, scales=1.0):
    """Returns x and y for a given angle.

    Parameters
    ----------
    angles: np.ndarray
        The angular coordinates
    scales: np.ndarray
        The radial coordinates

    Returns
    -------
    np.ndarray:
        The x-coordinate and y-coordinates

    """
    return np.stack([scales * np.cos(angles), scales * np.sin(angles)], axis=-1)


def create_template_matrix(
    n_radial, n_angular, radius, in_cart=False, exp_lambda=1.0, shift_angular=True
):
    """Creates a template matrix with radius `radius` and `n_radial` radial- and `n_angular` angular coordinates.

    Parameters
    ----------
    n_radial: int
        The amount of radial coordinates
    n_angular: int
        The amount of angular coordinates
    radius: float
        The radius of the template
    in_cart: bool
        If True, then the template matrix contains cartesian coordinates
    exp_lambda: float
        Whether to sample more points closer to the origin than farther out. This lambda determines the strength
        of how non-uniform to sample.
    shift_angular: bool
        Whether to add half of angular step to every second row of template vertices.

    Returns
    -------
    np.ndarray
        A template matrix K with K[i, j] containing polar coordinates (radial, angular) of point (i. j)
    """

    coordinates = np.zeros((n_radial, n_angular, 2))
    for j in range(1, n_radial + 1):
        radial_coordinate = ((j / n_radial) ** exp_lambda) * radius
        for k in range(1, n_angular + 1):
            angular_coordinate = (
                2 * k * np.pi + int(shift_angular) * np.mod(j, 2) * np.pi
            ) / n_angular

            coordinates[j - 1, k - 1, 0] = radial_coordinate
            coordinates[j - 1, k - 1, 1] = angular_coordinate

    if in_cart:
        for rc in range(coordinates.shape[0]):
            for ac in range(coordinates.shape[1]):
                coordinates[rc, ac] = polar_to_cart(
                    coordinates[rc, ac, 1], coordinates[rc, ac, 0]
                )

    return coordinates


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
    if denominator == 0:
        denominator += sys.float_info.min
    point_2_weight = (dot11 * dot02 - dot01 * dot12) / denominator
    point_1_weight = (dot00 * dot12 - dot01 * dot02) / denominator
    point_0_weight = 1 - point_2_weight - point_1_weight

    is_inside_triangle = (
        point_2_weight > 0
        and point_1_weight > 0
        and point_2_weight + point_1_weight <= 1
    )

    return (point_0_weight, point_1_weight, point_2_weight), is_inside_triangle


def interpolation(query_point, gpc_triangles, gpc_triangles_node_indices):
    """Interpolates a query point within a GPC-system

    Parameters
    ----------
    query_point: np.ndarray
        A query point in cartesian coordinates
    gpc_triangles: np.ndarray
        The triangles contained in the GPC-system
    gpc_triangles_node_indices: np.ndarray
        The indices of the triangles contained in the GPC-system

    Returns
    -------
    (np.ndarray, np.ndarray):
        The first returned array contains the barycentric coordinates. The second returned array contains the
        indices of the vertices to which the barycentric coordinates belong. In case of 'vertex' not falling into
        any triangle, two zero arrays are returned. I.e. there will be no feature taken into consideration for this
        template vertex.
    """
    for idx, triangle in enumerate(gpc_triangles):
        b_coordinates, lies_within = compute_barycentric(query_point, triangle)
        if lies_within:
            return np.stack([b_coordinates, gpc_triangles_node_indices[idx]], axis=-1)
    return np.stack([np.array([0.0, 0.0, 0.0]), np.array([0, 0, 0])], axis=-1)
