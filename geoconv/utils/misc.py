from geoconv.preprocessing.barycentric_coords import determine_gpc_triangles, barycentric_coordinates_kernel
from geoconv.preprocessing.discrete_gpc import compute_gpc_systems

import numpy as np
import trimesh


def shuffle_mesh_vertices(object_mesh):
    shuffled_node_indices = np.arange(object_mesh.vertices.shape[0])
    np.random.shuffle(shuffled_node_indices)
    object_mesh_vertices = np.copy(object_mesh.vertices)[shuffled_node_indices]
    object_mesh_faces = np.copy(object_mesh.faces)
    for face in object_mesh_faces:
        face[0] = np.where(shuffled_node_indices == face[0])[0]
        face[1] = np.where(shuffled_node_indices == face[1])[0]
        face[2] = np.where(shuffled_node_indices == face[2])[0]
    return trimesh.Trimesh(vertices=object_mesh_vertices, faces=object_mesh_faces), shuffled_node_indices


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


def exp_map(radial_c, angular_c, center_vertex, mesh):
    """Maps a point located in the tangent plane of a mesh vertex onto the mesh surface

    Parameters
    ----------
    radial_c: float
        The radial coordinate of the point to map onto the surface
    angular_c: float
        The angular coordinate of the point to map onto the surface
    center_vertex: int
        The index for the mesh vertex in which we consider the tangent plane
    mesh: trimesh.Trimesh
        The triangle mesh

    Returns
    -------
    np.ndarray:
        An array that contains the 3D cartesian coordinates for the mapped point

    """
    gpc_system = compute_gpc_systems(mesh)[center_vertex]
    contained_gpc_triangles, contained_gpc_faces = determine_gpc_triangles(mesh, gpc_system)

    x, y = polar_to_cart(angular_c, scale=radial_c)
    bary_coord = barycentric_coordinates_kernel(np.array([[[x, y]]]), contained_gpc_triangles, contained_gpc_faces)

    mesh_vertices = np.asarray(mesh.vertices[bary_coord[0, 0, :3, 0].astype(np.int16)])

    return mesh_vertices @ bary_coord[:, :, :, 1][0, 0]


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
