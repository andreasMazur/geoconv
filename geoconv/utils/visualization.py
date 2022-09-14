from matplotlib import pyplot as plt

import trimesh
import numpy as np


def visualize_gpc_on_mesh(center_vertex, radial_coordinates, angular_coordinates, object_mesh):
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


def visualize_gpc_in_coordinate_system(radial_coordinates, angular_coordinates, object_mesh, kernel=None):
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
