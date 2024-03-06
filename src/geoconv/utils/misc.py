from geoconv.preprocessing.barycentric_coordinates import polar_to_cart

from tqdm import tqdm
from scipy.linalg import blas

import pygeodesic.geodesic as geodesic
import numpy as np
import trimesh


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
    # 3.) Return sorted edge and corresponding faces
    return edge, considered_faces


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


def normalize_mesh(mesh, geodesic_diameter=None):
    """Center mesh and scale x, y and z dimension with '1/geodesic diameter'.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The triangle mesh, that shall be normalized
    geodesic_diameter: float
        The geodesic diameter. If not provided, this function will compute the geodesic diameter.

    Returns
    -------
    (trimesh.Trimesh, float):
        The normalized mesh and the geodesic diameter, with which the mesh was normalized
    """
    # Center mesh
    for dim in range(3):
        mesh.vertices[:, dim] = mesh.vertices[:, dim] - mesh.vertices[:, dim].mean()

    # Determine geodesic diameter
    if geodesic_diameter is None:
        distance_matrix, geodesic_diameter = compute_geodesic_diameter(mesh)

    # Scale mesh
    for dim in range(3):
        mesh.vertices[:, dim] = mesh.vertices[:, dim] * (1 / geodesic_diameter)
    print(f"-> Normalized with geodesic diameter: {geodesic_diameter}")

    return mesh, geodesic_diameter


def compute_geodesic_diameter(mesh):
    """Computes the geodesic diameter of a mesh.

    In case the mesh contains a pair of vertices which are not connected by a path, this
    function returns the largest geodesic distance that has been seen as the geodesic diameter.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The triangle mesh, for which the geodesic diameter shall be calculated.

    Returns
    -------
    (np.array, float):
        The distance matrix between all vertices of the mesh and the geodesic diameter of the mesh.
    """
    n_vertices = mesh.vertices.shape[0]
    distance_matrix = np.zeros((n_vertices, n_vertices))
    geoalg = geodesic.PyGeodesicAlgorithmExact(mesh.vertices, mesh.faces)
    for sp in tqdm(range(n_vertices), postfix=f"Calculating geodesic diameter.."):
        distances, _ = geoalg.geodesicDistances([sp], None)
        distance_matrix[sp] = distances
    return distance_matrix, distance_matrix[distance_matrix != np.inf].max()


def gpc_systems_into_cart(gpc_systems):
    """Translates the geodesic polar coordinates of given GPC-systems into cartesian

    Parameters
    ----------
    gpc_systems: np.ndarray
        A 3D-array containing all GPC-systems which shall be translated

    Returns
    -------
    np.ndarray:
        The same GPC-systems but in cartesian coordinates
    """
    gpc_systems_cart = np.copy(gpc_systems)
    return polar_to_cart(gpc_systems_cart[:, :, 1], gpc_systems_cart[:, :, 0])


def reconstruct_template(gpc_system, b_coordinates):
    """Reconstructs the template vertices with barycentric coordinates

    Parameters
    ----------
    gpc_system: np.ndarray
        A 2D-array that describes a GPC-system. I.e. 'gpc_system[i]' contains the
        geodesic polar coordinates (radial, angle) for the i-th vertex of the underlying
        object mesh.
    b_coordinates: np.ndarray
        Contains the barycentric coordinates from which the template shall be reconstructed.
    Returns
    -------
    np.ndarray:
        Cartesian template coordinates in the same format as returned by 'create_template_matrix'.

    """

    reconstructed_template = np.zeros((b_coordinates.shape[0], b_coordinates.shape[1], 2))
    for rc in range(b_coordinates.shape[0]):
        for ac in range(b_coordinates.shape[1]):
            # Get vertices
            vertex_indices = b_coordinates[rc, ac, :, 0].astype(np.int16)
            vertices = [(gpc_system[vertex_indices[idx], 0], gpc_system[vertex_indices[idx], 1]) for idx in range(3)]
            vertices = np.array([polar_to_cart(angles=y, scales=x) for x, y in vertices])

            # Interpolate vertices
            weights = b_coordinates[rc, ac, :, 1]
            reconstructed_template[rc, ac] = vertices.T @ weights
    return reconstructed_template


def shuffle_mesh_vertices(mesh, given_shuffle=None):
    """Shuffles the vertices of the mesh

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The mesh from which you want to shuffle the vertices
    given_shuffle: np.ndarray
        A given shuffle of the vertices

    Returns
    -------
    (trimesh.Trimesh, np.ndarray, np.ndarray)
        The same mesh but with a different vertices order. Additionally, two arrays are returned. Both contain vertex
        indices. Given a vertex index 'idx', it holds that:

        mesh.vertices[idx] == shuffled_mesh.vertices[shuffle_map[idx]] == mesh.vertices[ground_truth[shuffle_map[idx]]]
    """
    ground_truth = np.arange(mesh.vertices.shape[0])
    if given_shuffle is None:
        np.random.shuffle(ground_truth)
    else:
        ground_truth = np.copy(given_shuffle)
    mesh_vertices = np.copy(mesh.vertices)[ground_truth]

    shuffle_map = []
    for vertex_idx in range(mesh.vertices.shape[0]):
        shuffle_map.append(np.where(ground_truth == vertex_idx)[0])
    shuffle_map = np.array(shuffle_map).flatten()

    mesh_faces = np.copy(mesh.faces)
    for face_idx in range(mesh.faces.shape[0]):
        mesh_faces[face_idx, 0] = shuffle_map[mesh.faces[face_idx, 0]]
        mesh_faces[face_idx, 1] = shuffle_map[mesh.faces[face_idx, 1]]
        mesh_faces[face_idx, 2] = shuffle_map[mesh.faces[face_idx, 2]]

    shuffled_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)

    return shuffled_mesh, shuffle_map, ground_truth


def get_included_faces(object_mesh, gpc_system):
    """Retrieves face indices from GPC-system

    Parameters
    ----------
    object_mesh: trimesh.Trimesh
        The object mesh
    gpc_system: np.ndarray
        The considered GPC-system

    Returns
    -------
    list:
        The list of face IDs which are included in the GPC-system
    """
    included_face_ids = []

    # Determine vertex IDs that are included in the GPC-system
    gpc_vertex_ids = np.arange(gpc_system.shape[0])[gpc_system[:, 0] != np.inf]

    # Determine what faces are entirely contained within the GPC-system
    for face_id, face in enumerate(object_mesh.faces):
        counter = 0
        for vertex_id in face:
            counter = counter + 1 if vertex_id in gpc_vertex_ids else counter
        if counter == 3:
            included_face_ids.append(face_id)

    return included_face_ids


def get_points_from_polygons(polygons):
    """Returns the unique set of points given in a set of polygons

    Parameters
    ----------
    polygons: np.ndarray
        Set of polygons from which the set of unique points will be returned

    Returns
    -------
    np.ndarray
        The set of unique points
    """
    return np.unique(polygons.reshape((-1, 2)), axis=0)


def find_largest_one_hop_dist(object_mesh):
    """Finds the largest Euclidean distance from center vertex to a one-hop neighbor in a triangle mesh

    Returns
    -------
    float:
        The largest initialization distance from a center-vertex to a one-hop neighbor in the triangle mesh
    """
    all_edges = object_mesh.vertices[object_mesh.edges]
    return np.linalg.norm(all_edges[:, 0, :] - all_edges[:, 1, :], axis=-1).max()
