from geoconv.preprocessing.barycentric_coordinates import polar_to_cart, determine_gpc_triangles
from geoconv.preprocessing.discrete_gpc import local_gpc

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon

import trimesh
import numpy as np

from geoconv.utils.misc import get_points_from_polygons


def draw_triangles(triangles, points=None):
    """Draws a single triangle and optionally a point in 2D space.

    Parameters
    ----------
    triangles: np.ndarray
        The triangles in cartesian coordinates
    points: np.ndarray
        Points that can optionally also be visualized
    """
    fig, ax = plt.subplots(1, 1)
    for tri in triangles:
        polygon = Polygon(tri, alpha=.4, edgecolor="red")
        ax.add_patch(polygon)

    if points is not None:
        for point in points:
            ax.scatter(point[0], point[1])

    if points is None:
        ax.set_xlim(triangles[:, :, 0].min(), triangles[:, :, 0].max())
        ax.set_ylim(triangles[:, :, 1].min(), triangles[:, :, 1].max())

    plt.show()


def draw_princeton_benchmark(paths, labels, figure_name):
    """Visualizes the Princeton benchmark plots

    Parameters
    ----------
    paths: list
        A list of paths to the npy-files
    labels: list
        A list of labels in order with the paths
    figure_name: str
        The name of the figure
    """
    plt.yticks(np.linspace(0., 1., num=11))
    for idx, path in enumerate(paths):
        arr = np.load(path)
        label_name = f"{labels[idx]} -" \
                     f" Mean Error: {arr[:, 1].mean():.3f} -" \
                     f" 80% at: {arr[np.argmax(arr[:, 0] >= .8)][1]:.3f}"
        plt.plot(arr[:, 1], arr[:, 0], label=label_name)

    plt.title("Princeton Benchmark")
    plt.xlabel("geodesic error")
    plt.ylabel("% correct correspondences")
    plt.legend()
    plt.grid()
    plt.savefig(f"{figure_name}.svg")
    plt.show()


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
    point_cloud_1 = trimesh.points.PointCloud(object_mesh.vertices, colors=colors)
    to_visualize = [object_mesh, point_cloud_1]
    trimesh.Scene(to_visualize).show()

    # Visualize angular coordinates
    colors = trimesh.visual.interpolate(angular_coordinates, color_map="YlGn")
    colors[center_vertex] = np.array([255, 255, 0, 255])
    point_cloud_2 = trimesh.points.PointCloud(object_mesh.vertices, colors=colors)
    to_visualize.append(point_cloud_2)
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


def draw_gpc_triangles(object_mesh,
                       center_vertex,
                       u_max=.04,
                       kernel_matrix=None,
                       alpha=.4,
                       edge_color="red",
                       scatter_color="green",
                       use_c=True,
                       plot=True,
                       title=""):
    """Draws the triangles of a local GPC-system.

    Parameters
    ----------
    object_mesh: trimesh.Trimesh
        The object mesh on which to compute the GPC-system.
    center_vertex: int
        The center vertex of the GPC-system which shall be visualized.
    u_max: float
        The max-radius of the GPC-system
    kernel_matrix: np.ndarray
        A 3D-array that describes kernel vertices in cartesian coordinates. If 'None' is passed
        no kernel vertices will be visualized.
    alpha: float
        The opacity of the polygons
    edge_color: str
        The color for the triangle edges
    scatter_color: str
        The color for the kernel vertices (in case a kernel is given)
    use_c: bool
        Whether to use the C-extension to compute the GPC-system.
    plot: bool
        Whether to immediately plot
    title: str
        The title of the plot
    """
    radial, angular, _ = local_gpc(center_vertex, u_max=u_max, object_mesh=object_mesh, use_c=use_c)
    gpc_system = np.stack([radial, angular], axis=1)
    contained_gpc_triangles, _ = determine_gpc_triangles(
        object_mesh, gpc_system
    )
    for tri_idx in range(contained_gpc_triangles.shape[0]):
        for point_idx in range(contained_gpc_triangles.shape[1]):
            rho, theta = contained_gpc_triangles[tri_idx, point_idx]
            contained_gpc_triangles[tri_idx, point_idx] = polar_to_cart(theta, scale=rho)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlim([contained_gpc_triangles.min(), contained_gpc_triangles.max()])
    ax.set_ylim([contained_gpc_triangles.min(), contained_gpc_triangles.max()])
    polygons = PolyCollection(contained_gpc_triangles, alpha=alpha, edgecolors=edge_color)
    ax.add_collection(polygons)

    points = get_points_from_polygons(contained_gpc_triangles)
    ax.scatter(points[:, 0], points[:, 1], color="red")

    if kernel_matrix is not None:
        for radial_idx in range(kernel_matrix.shape[0]):
            ax.scatter(kernel_matrix[radial_idx, :, 0], kernel_matrix[radial_idx, :, 1], color=scatter_color)

    if plot:
        plt.show()

    return gpc_system


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
