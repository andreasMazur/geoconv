from geoconv.utils.misc import compute_vector_angle

import numpy as np
import c_extension


def dgpc_angle_update(vector_i, vector_j, vector_k, theta_j, theta_k):
    """Computes the angular update for the DGPC-algorithm.

    Parameters
    ----------
    vector_i: np.ndarray
        3D-coordinates of vertex i, whose geodesic polar coordinate shall be computed.
    vector_j: np.ndarray
        3D-coordinates of vertex j.
    vector_k: np.ndarray
        3D-coordinates of vertex k.
    theta_j: float
        Angular direction to vertex j.
    theta_k: float
        Angular direction to vertex k.

    Returns
    -------
    float:
        The new angular direction to vertex i.
    """
    # Angles w.r.t. between vector and vector having the largest angle
    phi_kj = compute_vector_angle(vector_k, vector_j, None)
    phi_ij = compute_vector_angle(vector_i, vector_j, None)
    if theta_k <= theta_j:
        if theta_j - theta_k >= np.pi:
            theta_k = theta_k + 2 * np.pi
    else:
        if theta_k - theta_j >= np.pi:
            theta_j = theta_j + 2 * np.pi
    alpha = phi_ij / phi_kj

    # Pay attention to 0-2pi-discontinuity
    return np.fmod((1 - alpha) * theta_j + alpha * theta_k, 2 * np.pi)


def dgpc_update_step_python(vertex_i_3d, vertex_j_3d, vertex_k_3d, u_j, u_k, theta_j, theta_k):
    """The DGPC-algorithm update step for one unknown vertex.

    Parameters
    ----------
    vertex_i_3d: np.ndarray
        3D-coordinates of vertex i, whose geodesic polar coordinate shall be computed.
    vertex_j_3d: np.ndarray
        3D-coordinates of vertex j.
    vertex_k_3d: np.ndarray
        3D-coordinates of vertex k.
    u_j: float
        Geodesic distance to vertex j.
    u_k: float
        Geodesic distance to vertex k.
    theta_j: float
        Angular direction to vertex j.
    theta_k: float
        Angular direction to vertex k.

    Returns
    -------
    (float, float):
        The new geodesic distance and angular direction to vertex i.
    """
    # Compute x_j and x_k
    e_j = vertex_j_3d - vertex_i_3d
    e_k = vertex_k_3d - vertex_i_3d
    e_kj = e_k - e_j
    # e_kj_sqnrm = np.einsum("i,i->", e_kj, e_kj)
    e_kj_norm = np.linalg.norm(e_kj)
    e_kj_sqnrm = np.square(e_kj_norm)
    A = np.linalg.norm(np.cross(e_j, e_k))

    # Variant 1:
    # radicand = (e_kj_sqnrm - (u_j - u_k) ** 2) * ((u_j + u_k) ** 2 - e_kj_sqnrm)
    # radicand = 0. if -1e12 < radicand < 0. else radicand

    # Variant 2:
    a, b, c = np.sort([u_j, u_k, e_kj_norm])
    radicand = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    radicand = 0. if -1e-3 < radicand < 0. else radicand

    if radicand >= 0:
        H = np.sqrt(radicand)
        denominator = 2. * A * e_kj_sqnrm
        if denominator <= 0.:
            return np.inf, -1.
        else:
            x_j = A * (e_kj_sqnrm + u_k ** 2 - u_j ** 2) + np.einsum("i,i->", e_k, e_kj) * H
            x_k = A * (e_kj_sqnrm + u_j ** 2 - u_k ** 2) - np.einsum("i,i->", e_j, e_kj) * H
            x_j, x_k = np.array([x_j, x_k]) / denominator
            x_j = 0. if -1e-3 < x_j < 0. else x_j
            x_k = 0. if -1e-3 < x_k < 0. else x_k

            if x_k < 0 or x_j < 0:
                return np.inf, -1.
            else:
                linear_comb = x_j * e_j + x_k * e_k
                s = vertex_i_3d + x_j * e_j + x_k * e_k
                theta_i = dgpc_angle_update(
                    vector_i=vertex_i_3d - s,
                    vector_j=vertex_j_3d - s,
                    vector_k=vertex_k_3d - s,
                    theta_j=theta_j,
                    theta_k=theta_k
                )
                if theta_i == -1.:
                    return np.inf, -1.
                return np.linalg.norm(linear_comb), theta_i
    else:
        return np.inf, -1.


def dgpc_update_step(vertex_i_3d, vertex_j_3d, vertex_k_3d, u_j, u_k, theta_j, theta_k, use_c=True):
    """A wrapper function for the DGPC-algorithm update step for one unknown vertex.

    Parameters
    ----------
    vertex_i_3d: np.ndarray
        3D-coordinates of vertex i, whose geodesic polar coordinate shall be computed.
    vertex_j_3d: np.ndarray
        3D-coordinates of vertex j.
    vertex_k_3d: np.ndarray
        3D-coordinates of vertex k.
    u_j: float
        Geodesic distance to vertex j.
    u_k: float
        Geodesic distance to vertex k.
    theta_j: float
        Angular direction to vertex j.
    theta_k: float
        Angular direction to vertex k.
    use_c: bool
        Whether to use the C-extension instead of the Python implementation.

    Returns
    -------
    (float, float):
        The new geodesic distance and angular direction to vertex i.
    """
    if use_c:
        rotation_axis = np.zeros((3,), dtype=np.float64)  # Deprecated parameter
        result = np.array([0.0, 0.0])
        c_extension.compute_dist_and_dir(
            result,
            vertex_i_3d,
            vertex_j_3d,
            vertex_k_3d,
            u_j,
            u_k,
            theta_j,
            theta_k,
            rotation_axis
        )
        u_i, theta_i = result
    else:
        u_i, theta_i = dgpc_update_step_python(vertex_i_3d, vertex_j_3d, vertex_k_3d, u_j, u_k, theta_j, theta_k)
    return u_i, theta_i
