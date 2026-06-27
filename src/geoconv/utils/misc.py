from io import BytesIO
from scipy.linalg import blas
from tqdm import tqdm

import numpy as np
import trimesh


def angle_distance(theta_max, theta_min):
    """Compute the shortest angular distance between two angles

    Parameters
    ----------
    theta_max: float
        The first angle
    theta_min: float
        The second angle

    Returns
    -------
    float:
        The shortest angular distance between the two angles
    """
    return np.minimum(theta_max - theta_min, theta_min + 2.0 * np.pi - theta_max)


def normal_pdf(mean_rho, mean_theta, var_rho, var_theta, rho, theta):
    """Normal probability distribution for geodesic polar coordinates

    Parameters
    ----------
    mean_rho: float
        Mean radial distance of the normal
    mean_theta: float
        Mean angle for the Gaussian
    var_rho: float
        Mean radial distance variance of the kernel vertices
    var_theta: float
        Mean angle variance of the kernel vertices
    rho: float
        Radial coordinate of the interpolation point that shall be weighted
    theta: float
        Angular coordinate of the interpolation point that shall be weighted

    Returns
    -------
    float:
        The weight for the interpolation point (rho, theta)
    """
    norm_coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * var_rho * var_theta)
    max_angle = np.maximum(mean_theta, theta)
    min_angle = np.minimum(mean_theta, theta)
    delta_angle = angle_distance(max_angle, min_angle)
    vec = np.array([rho - mean_rho, delta_angle])
    mat = np.array([[1 / var_rho, 0], [0, 1 / var_theta]])
    exp = np.exp(-(1 / 2) * vec.T @ mat @ vec)
    return norm_coefficient * exp


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


def get_faces_of_edge(edge, object_mesh):
    """Determine both faces of a given edge

    Parameters
    ----------
    edge: np.ndarray
        The edge for which the faces shall be returned.
    object_mesh: trimesh.Trimesh
        The underlying mesh.
    """
    edge = np.sort(edge)
    # 1.) Get the edge index of `sorted_edge` "in both ways", i.e. two indices for `sorted_edge`
    edge_indices = object_mesh.edges_sorted == edge
    edge_indices = np.where(np.logical_and(edge_indices[:, 0], edge_indices[:, 1]))
    # 2.) Get faces of `sorted_edge` by retrieving `face_indices` for the found `edge_indices`
    face_indices = object_mesh.edges_face[edge_indices]
    considered_faces = object_mesh.faces[face_indices]
    # 3.) Return faces of sorted edge
    return np.array(considered_faces)


def remove_nme(mesh):
    """Removes non-manifold edges by removing all their faces.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The triangle mesh.

    Returns
    -------
    trimesh.Trimesh:
        The mesh without non-manifold edges.
    """
    # Check if non-manifold edges exist
    non_manifold_edges = np.asarray(mesh.as_open3d.get_non_manifold_edges())
    if non_manifold_edges.shape[0] > 0:
        # Compute mask that removes non-manifold edges and all their faces
        face_mask = np.full(mesh.faces.shape[0], True)
        for edge in tqdm(non_manifold_edges, desc="Removing non-manifold edges.."):
            sorted_edge = np.sort(edge)
            edge_faces = get_faces_of_edge(sorted_edge, mesh)
            for edge_f in edge_faces:
                update_mask = np.logical_not((edge_f == mesh.faces).all(axis=-1))
                face_mask = np.logical_and(face_mask, update_mask)
        # Remove non-manifold edges and faces with mask
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[face_mask])
    return mesh


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


def repair_mesh(mesh):
    """Merges very close vertices and removes degenerate faces (faces without 3 unique vertices).

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The mesh to validate.

    Returns
    -------
    trimesh.Trimesh:
        The repaired mesh.
    """
    # 'merge_vertices'
    # mesh.merge_vertices(merge_tex=True, merge_norm=True)  # (does not update vertex_adjacency_graph)
    # Remove degenerate faces
    # mesh.update_faces(mesh.nondegenerate_faces())  # (does not update vertex_adjacency_graph)
    # Observed cases in which loaded mesh 'trimesh.load_mesh(...)' has less vertices than this:
    # trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True, validate=True)

    # Merges vertices
    loaded_mesh = trimesh.load_mesh(
        BytesIO(mesh.export(file_type="stl")), file_type="stl"
    )

    # Repairs faces
    mesh = trimesh.Trimesh(
        vertices=loaded_mesh.vertices,
        faces=loaded_mesh.faces,
        process=True,
        validate=True,
    )

    return mesh


def compute_distance_matrix(vertices):
    """Computes the Euclidean distance between given vertices.

    Parameters
    ----------
    vertices: np.ndarray
        The vertices to compute the distance between.

    Returns
    -------
    np.ndarray:
        A square distance matrix for the given vertices.
    """
    norm = np.einsum("ij,ij->i", vertices, vertices)
    norm = (
        np.reshape(norm, (-1, 1))
        - 2 * np.einsum("ik,jk->ij", vertices, vertices)
        + np.reshape(norm, (1, -1))
    )
    norm[np.isnan(np.sqrt(norm))] = 0.0

    return np.sqrt(norm)


def compute_sub_distance_matrix(vertices, indices):
    """Computes the distances for each vertex whose index is in 'indices'.

    Parameters
    ----------
    vertices: np.ndarray
        All vertices towards which distances shall be computed.
    indices: np.ndarray
        The indices for the vertices from which distances shall be computed.

    Returns
    -------
    np.ndarray:
        A sub-distance matrix for the given vertices.
    """
    return np.linalg.norm(vertices[indices][:, None] - vertices[None, ...], axis=-1, ord=2)
