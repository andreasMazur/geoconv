from geoconv.preprocessing.barycentric_coordinates import polar_to_cart
from geoconv.preprocessing.discrete_gpc import initialize_neighborhood

from tqdm import tqdm

import pygeodesic.geodesic as geodesic
import numpy as np
import trimesh


def normalize_mesh(mesh):
    """Center mesh and scale x, y and z dimension with '1/geodesic diameter'.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The triangle mesh, that shall be normalized

    Returns
    -------
    trimesh.Trimesh:
        The normalized mesh
    """
    # Center mesh
    for dim in range(3):
        mesh.vertices[:, dim] = mesh.vertices[:, dim] - mesh.vertices[:, dim].mean()

    # Determine geodesic distances
    n_vertices = mesh.vertices.shape[0]
    distance_matrix = np.zeros((n_vertices, n_vertices))
    geoalg = geodesic.PyGeodesicAlgorithmExact(mesh.vertices, mesh.faces)
    for sp in tqdm(range(n_vertices), postfix=f"Normalizing mesh.."):
        distances, _ = geoalg.geodesicDistances([sp], None)
        distance_matrix[sp] = distances

    # Scale mesh
    geodesic_diameter = distance_matrix.max()
    for dim in range(3):
        mesh.vertices[:, dim] = mesh.vertices[:, dim] * (1 / geodesic_diameter)
    print(f"-> Normalized with geodesic diameter: {geodesic_diameter}")

    return mesh


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
    for gpc_system_idx in tqdm(range(gpc_systems_cart.shape[0]), postfix="Translating GPC-coordinates into cartesian"):
        for vertex_idx in range(gpc_systems_cart.shape[1]):
            gpc_systems_cart[gpc_system_idx, vertex_idx] = polar_to_cart(
                gpc_systems_cart[gpc_system_idx, vertex_idx, 1], gpc_systems_cart[gpc_system_idx, vertex_idx, 0]
            )
    return gpc_systems_cart


def reconstruct_kernel(gpc_system, b_coordinates):
    """Reconstructs the kernel vertices with barycentric coordinates

    Parameters
    ----------
    gpc_system: np.ndarray
        A 2D-array that describes a GPC-system. I.e. 'gpc_system[i]' contains the
        geodesic polar coordinates (radial, angle) for the i-th vertex of the underlying
        object mesh.
    b_coordinates: np.ndarray
        Contains the barycentric coordinates from which the kernel shall be reconstructed.
    Returns
    -------
    np.ndarray:
        Cartesian kernel coordinates in the same format as returned by 'create_kernel_matrix'.

    """

    reconstructed_kernel = np.zeros((b_coordinates.shape[0], b_coordinates.shape[1], 2))
    for rc in range(b_coordinates.shape[0]):
        for ac in range(b_coordinates.shape[1]):
            # Get vertices
            vertex_indices = b_coordinates[rc, ac, :, 0].astype(np.int16)
            vertices = [(gpc_system[vertex_indices[idx], 0], gpc_system[vertex_indices[idx], 1]) for idx in range(3)]
            vertices = np.array([polar_to_cart(y, x) for x, y in vertices])

            # Interpolate vertices
            weights = b_coordinates[rc, ac, :, 1]
            reconstructed_kernel[rc, ac] = vertices.T @ weights
    return reconstructed_kernel


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


def find_smallest_radius(object_mesh, use_c=True):
    """Finds the largest Euclidean distance from center vertex to a one-hop neighbor in a triangle mesh

    The initialization of the algorithm that computes the GPC-systems cannot
    ensure that the radial coordinates of the center-vertex's one-hop neighbors
    are smaller than 'u_max'.

    This function returns the largest radial coordinate that has been seen during
    initialization.

    Returns
    -------
    float:
        The largest initialization distance from a center-vertex to a one-hop neighbor in the triangle mesh
    """
    largest_radial_c = .0
    for source_point in tqdm(
        range(object_mesh.vertices.shape[0]), postfix="Checking for largest initial radial distances"
    ):
        u = np.full((object_mesh.vertices.shape[0],), np.inf)
        theta = np.full((object_mesh.vertices.shape[0],), -1.0)
        u, theta, source_point_neighbors, rotation_axis = initialize_neighborhood(
            source_point, u, theta, object_mesh, use_c
        )
        u[source_point] = .0
        gpc_largest_radial_c = np.array([x for x in u if not np.isinf(x)]).max()
        largest_radial_c = largest_radial_c if largest_radial_c >= gpc_largest_radial_c else gpc_largest_radial_c
    return largest_radial_c
