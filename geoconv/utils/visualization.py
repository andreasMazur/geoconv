from geoconv.preprocessing.barycentric_coords import determine_gpc_triangles, get_points_from_polygons, polar_to_cart
from geoconv.preprocessing.discrete_gpc import local_gpc

from matplotlib.collections import PolyCollection
from matplotlib import pyplot as plt

import trimesh
import numpy as np


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


def draw_correspondences(query_mesh, reference_mesh, ground_truth, predictions=None):
    """TODO: Visualizes ground truth and predicted correspondences

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


def draw_gpc_triangles(object_mesh, center_vertex, u_max=.04, kernel=None, alpha=.4, edge_color="red", use_c=True):
    """Draws the triangles of a local GPC-system.

    Parameters
    ----------
    object_mesh: trimesh.Trimesh
        The object mesh on which to compute the GPC-system.
    center_vertex: int
        The center vertex of the GPC-system which shall be visualized.
    u_max: float
        The max-radius of the GPC-system
    kernel: np.ndarray
        A 3D-array that describes kernel vertices in cartesian coordinates. If 'None' is passed
        no kernel vertices will be visualized.
    alpha: float
        The opacity of the polygons
    edge_color: str
        The color for the triangle edges
    use_c: bool
        Whether to use the C-extension to compute the GPC-system.
    """
    radial, angular, _ = local_gpc(center_vertex, u_max=u_max, object_mesh=object_mesh, use_c=use_c)
    contained_gpc_triangles, _ = determine_gpc_triangles(
        object_mesh, np.stack([radial, angular], axis=1)
    )
    for tri_idx in range(contained_gpc_triangles.shape[0]):
        for point_idx in range(contained_gpc_triangles.shape[1]):
            rho, theta = contained_gpc_triangles[tri_idx, point_idx]
            contained_gpc_triangles[tri_idx, point_idx] = polar_to_cart(theta, scale=rho)

    fig, ax = plt.subplots()
    ax.set_xlim([contained_gpc_triangles.min(), contained_gpc_triangles.max()])
    ax.set_ylim([contained_gpc_triangles.min(), contained_gpc_triangles.max()])
    polygons = PolyCollection(contained_gpc_triangles, alpha=alpha, edgecolors=edge_color)
    ax.add_collection(polygons)

    points = get_points_from_polygons(contained_gpc_triangles)
    ax.scatter(points[:, 0], points[:, 1], color="red")

    if kernel is not None:
        for radial_idx in range(kernel.shape[0]):
            ax.scatter(kernel[radial_idx, :, 0], kernel[radial_idx, :, 1], color="green")

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
