from matplotlib import pyplot as plt

import trimesh
import numpy as np


def draw_correspondences(query_mesh, reference_mesh, ground_truth, predictions=None):
    """Visualizes ground truth and predicted correspondences

    Parameters
    ----------
    query_mesh: trimesh.Trimesh
        The query mesh
    reference_mesh: trimesh.Trimesh
        The reference mesh
    ground_truth: np.ndarray
        The ground truth correspondences (vertex indices) in a 1D array
    predictions: np.ndarray
        The predicted correspondences (vertex indices) in a 1D array
    """

    # Assign ground truth colors
    colors = trimesh.visual.interpolate(np.arange(reference_mesh.vertices.shape[0]), color_map="gist_ncar")
    pc_t = trimesh.points.PointCloud(reference_mesh.vertices, colors=colors)

    # Color query mesh according to the found correspondences
    colors_query = pc_t.colors[ground_truth]
    pc_q = trimesh.points.PointCloud(query_mesh.vertices, colors=colors_query)

    # pc_t.vertices = pc_t.vertices - pc_t.centroid
    # pc_q.vertices = pc_q.vertices - pc_q.centroid
    # m = np.concatenate([pc_t.vertices, pc_q.vertices]).mean()
    # v = np.concatenate([pc_t.vertices, pc_q.vertices]).var()
    # pc_t.vertices = (pc_t.vertices - m) / v
    # pc_q.vertices = (pc_q.vertices - m) / v

    # Put target and query mesh side by side
    pc_q[:, 0] = pc_q[:, 0] + 1
    for x in [1, 2]:
        pc_t[:, x] = pc_t[:, x] - pc_t.vertices[:, x].min()
        pc_q[:, x] = pc_q[:, x] - pc_q.vertices[:, x].min()

    # TODO
    if predictions is not None:
        pass

    to_visualize = [pc_t, pc_q]
    trimesh.Scene(to_visualize).show()


def gpc_on_mesh(center_vertex, radial_coordinates, angular_coordinates, object_mesh):
    """Visualizes the radial and angular coordinates of a local GPC-system on an object mesh.

    This function first shows you the radial coordinates and then the angular coordinates.

    Parameters
    ----------
    center_vertex: int
        The index of the center vertex of the GPC-system.
    radial_coordinates: np.ndarray
        A 1-dimensional array containing the radial coordinates of the GPC-system.
    angular_coordinates: np.ndarray
        A 1-dimensional array containing the angular coordinates of the GPC-system.
    object_mesh: trimesh.Trimesh
        The object mesh.
    """
    object_mesh.visual.vertex_colors = [100, 100, 100, 100]

    # Visualize radial coordinates
    radial_coordinates[radial_coordinates == np.inf] = 0.0
    colors = trimesh.visual.interpolate(radial_coordinates, color_map="Reds")
    colors[center_vertex] = np.array([255, 255, 0, 255])
    point_cloud = trimesh.points.PointCloud(object_mesh.vertices, colors=colors)
    to_visualize = [object_mesh, point_cloud]
    trimesh.Scene(to_visualize).show()

    # Visualize angular coordinates
    colors = trimesh.visual.interpolate(angular_coordinates, color_map="YlGn")
    colors[center_vertex] = np.array([255, 255, 0, 255])
    point_cloud = trimesh.points.PointCloud(object_mesh.vertices, colors=colors)
    to_visualize = [object_mesh, point_cloud]
    trimesh.Scene(to_visualize).show()


def gpc_in_coordinate_system(radial_coordinates, angular_coordinates, object_mesh, kernel=None):
    """Plots a GPC-system in a polar coordinate system.

    Parameters
    ----------
    radial_coordinates: np.ndarray
        A 1-dimensional array containing the radial coordinates of the GPC-system.
    angular_coordinates: np.ndarray
        A 1-dimensional array containing the angular coordinates of the GPC-system.
    object_mesh: trimesh.Trimesh
        The object mesh.
    kernel: np.ndarray
        A kernel matrix K with K[i, j] containing polar coordinates (radial, angular) of point (i. j)
    """

    _, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Determine coordinates to plot
    mask = radial_coordinates != np.inf

    plotted_faces_ = []
    included_vertex_ids = np.where(mask)[0]
    for vertex in included_vertex_ids:
        # Get triangles in polar coordinate representation
        face_ids_of_vertex = np.where(object_mesh.faces == vertex)[0]
        vertex_faces_ = object_mesh.faces[face_ids_of_vertex]
        for face_id, face in zip(face_ids_of_vertex, vertex_faces_):
            if face_id in plotted_faces_:
                continue
            face_r = radial_coordinates[face]
            face_theta = angular_coordinates[face]

            # "Close" triangles
            face_r = np.concatenate([face_r, [face_r[0]]])
            face_theta = np.concatenate([face_theta, [face_theta[0]]])

            # Exclude triangles which are not entirely contained within GPC system
            if np.any(radial_coordinates[face] == np.inf) or np.any(angular_coordinates[face] == -1.):
                continue

            # Plot and remember
            ax.plot(face_theta, face_r)
            plotted_faces_.append(face_id)

    if kernel is not None:
        kernel = kernel.reshape((-1, 2))
        ax.plot(kernel[:, 1], kernel[:, 0], "bo")
    plt.show()


def vertices_in_coordinate_system(radial_coordinates, angular_coordinates):
    """Plots the vertices of a GPC-system in a polar coordinate system.

    Parameters
    ----------
    radial_coordinates: np.ndarray
        A 1-dimensional array containing the radial coordinates of the GPC-system.
    angular_coordinates: np.ndarray
        A 1-dimensional array containing the angular coordinates of the GPC-system.
    """

    # Plot vertices
    mask = radial_coordinates != np.inf
    _, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(angular_coordinates[mask], radial_coordinates[mask], "ro")
    plt.show()
