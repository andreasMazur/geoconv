from io import BytesIO

from geoconv.preprocessing.barycentric_coordinates import polar_to_cart

from scipy.linalg import blas

import numpy as np
import trimesh
import tempfile
import subprocess
import os
import pathlib

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
    return np.minimum(theta_max - theta_min, theta_min + 2. * np.pi - theta_max)

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
    """Center mesh and scale x, y and z dimension with '1/geodesic diameter' as well as merge vertices.

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

    # Merge vertices
    mesh = repair_mesh(mesh)

    return mesh, geodesic_diameter


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
    loaded_mesh = trimesh.load_mesh(BytesIO(mesh.export(file_type="stl")), file_type="stl")

    # Repairs faces
    mesh = trimesh.Trimesh(vertices=loaded_mesh.vertices, faces=loaded_mesh.faces, process=True, validate=True)

    return mesh


def compute_geodesic_diameter(mesh):
    """Computes the largest geodesic distance within a mesh.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The triangle mesh, for which the geodesic diameter shall be calculated.

    Returns
    -------
    (np.array, float):
        The distance matrix between all vertices of the mesh and the geodesic diameter of the mesh.
    """
    should_raise = False
    with tempfile.TemporaryDirectory(dir=".") as tempdir:
        np.save(f"{tempdir}/mesh_vertices.npy", mesh.vertices)
        np.save(f"{tempdir}/mesh_faces.npy", mesh.faces)

        current_env = os.environ.copy()
        proc = subprocess.run(
            [f"python", f"{pathlib.Path(__file__).parent.resolve()}/safe_pygeodesic.py", tempdir],
            env=current_env
        )
        if proc.returncode != 0:
            should_raise = True
        else:
            distance_matrix = np.load(f"{tempdir}/distance_matrix.npy")
    if should_raise:
        raise RuntimeError("Pygeodesic crashed processing!")

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


def find_largest_one_hop_dist(object_mesh):
    """Finds the largest Euclidean distance from center vertex to a one-hop neighbor in a triangle mesh

    Returns
    -------
    float:
        The largest initialization distance from a center-vertex to a one-hop neighbor in the triangle mesh
    """
    all_edges = object_mesh.vertices[object_mesh.edges]
    return np.linalg.norm(all_edges[:, 0, :] - all_edges[:, 1, :], axis=-1).max()


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
    norm = np.reshape(norm, (-1, 1)) - 2 * np.einsum("ik,jk->ij", vertices, vertices) + np.reshape(norm, (1, -1))
    norm[np.isnan(np.sqrt(norm))] = 0.

    return np.sqrt(norm)
