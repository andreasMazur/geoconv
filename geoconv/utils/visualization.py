from geoconv.preprocessing.barycentric_coordinates import polar_to_cart, determine_gpc_triangles, create_kernel_matrix
from geoconv.preprocessing.discrete_gpc import local_gpc
from geoconv.utils.misc import get_points_from_polygons

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon

import trimesh
import numpy as np
import matplotlib.cm as cm


def draw_interpolation_coefficients(icnn_layer, indices):
    """Wrapper method for 'draw_interpolation_coefficients_single_idx'

    Parameters
    ----------
    icnn_layer: geoconv.layers.conv_intrinsic.ConvIntrinsic
        The for which the interpolation coefficients shall be visualized
    indices: List[int]
        A list of index-tuple for accessing kernel vertices. I.e. indices[x] = (a, b) with K[a, b] = (rho, theta).
    """
    fig = plt.figure()
    rows = len(indices)
    ax_idx = 0
    for r_idx, a_idx in indices:
        ax_idx += 1
        axis_ic = fig.add_subplot(rows, 2, ax_idx)
        ax_idx += 1
        axis_kv = fig.add_subplot(rows, 2, ax_idx, projection="polar")
        draw_interpolation_coefficients_single_idx(icnn_layer, r_idx, a_idx, fig, axis_ic, axis_kv)
    fig.tight_layout()
    plt.show()


def draw_interpolation_coefficients_single_idx(icnn_layer, radial_idx, angular_idx, fig, axis_ic, axis_kv):
    """Visualizes the interpolation coefficients of the patch operator at a specific kernel vertex

    Parameters
    ----------
    icnn_layer: geoconv.layers.conv_intrinsic.ConvIntrinsic
        The for which the interpolation coefficients shall be visualized
    radial_idx: int
        The index of the radial coordinate from the kernel vertex for which we visualize the interpolation coefficients
    angular_idx: int
        The index of the angular coordinate from the kernel vertex for which we visualize the interpolation coefficients
    fig:
        The figure in which to plot the axes
    axis_ic:
        The axis on which to plot the interpolation coefficients matrix
    axis_kv:
        The axis on which to plot the weighted kernel vertices
    """

    # Get interpolation coefficients of the given layer: I[a, b] \in R^{n_radial * n_angular}
    weights = icnn_layer._interpolation_coefficients[radial_idx, angular_idx].numpy()
    kernel_size = icnn_layer._kernel_size
    kernel_matrix = icnn_layer._kernel_vertices.numpy()

    # Reshape vector into matrix: (n_radial * n_angular,) -> (n_radial, n_angular)
    # See 'ConvIntrinsic._configure_patch_operator()' for why it is stored as a vector.
    weights = weights.reshape(kernel_size)

    # Visualize interpolation coefficient matrix
    pos = axis_ic.matshow(weights, cmap="rainbow")
    fig.colorbar(pos, ax=axis_ic, fraction=0.046, pad=0.04)
    axis_ic.set_title(
        f"Interpolation Coefficients for: "
        f"({kernel_matrix[radial_idx, angular_idx, 0]:.3f}, {kernel_matrix[radial_idx, angular_idx, 1]:.3f})"
    )
    kernel_matrix = kernel_matrix.reshape((-1, 2))

    # TODO: For some reason, matplotlib only shows all labels from list[1:] and forgets about list[0]
    axis_ic.set_ylabel("Radial coordinate")
    axis_ic.set_yticklabels(["0"] + [f"{x:.3f}" for x in np.unique(kernel_matrix[:, 0])])

    axis_ic.set_xlabel("Angular coordinate (in radian)")
    axis_ic.set_xticklabels(["0"] + [f"{x:.3f}" for x in np.unique(kernel_matrix[:, 1])])

    # Visualize interpolation coefficients at kernel vertices
    axis_kv.scatter(kernel_matrix[:, 1], kernel_matrix[:, 0], c=cm.rainbow(weights.flatten()), s=150, edgecolor="black")
    axis_kv.set_title("Weights at Interpolation Points")
    axis_kv.grid(True)
    axis_kv.set_axisbelow(True)


def draw_correspondences(query_mesh, prediction, reference_mesh, color_map="Reds"):
    """Draw point correspondences between a query- and a reference mesh

    The point correspondence problem can be defined as labeling all vertices of a query
    mesh with the indices of the corresponding points in a reference mesh. See:

    > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://arxiv.org/abs/1501.06297)
    > Jonathan Masci and Davide Boscaini et al.

    Parameters
    ----------
    query_mesh: trimesh.Trimesh
        The mesh that contains the vertices, which you want to label
    prediction: np.ndarray
        The predicted labels for the vertices in the query mesh
    reference_mesh: trimesh.Trimesh
        The reference mesh
    color_map: str
        The used color map. Checkout 'matplotlib' for available color maps.
    """
    shift_dim = 0
    query_mesh.visual.vertex_colors = [100, 100, 100, 100]
    reference_mesh.visual.vertex_colors = [100, 100, 100, 100]

    ref_colors = trimesh.visual.interpolate(reference_mesh.vertices[:, shift_dim], color_map=color_map)
    reference_mesh_pc = trimesh.PointCloud(vertices=reference_mesh.vertices, colors=ref_colors)

    pred_colors = ref_colors[prediction]
    query_mesh.vertices[:, shift_dim] -= np.abs(
        query_mesh.vertices[:, shift_dim].min() - query_mesh.vertices[:, shift_dim].max()
    )
    query_mesh_pc = trimesh.PointCloud(vertices=query_mesh.vertices, colors=pred_colors)

    trimesh.Scene([query_mesh, query_mesh_pc, reference_mesh, reference_mesh_pc]).show()


def draw_princeton_benchmark(paths, labels, figure_name):
    """Visualizes the Princeton benchmark plots

    First, conduct the princeton benchmark. See 'geoconv.utils.princeton_benchmark'.

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


def draw_gpc_on_mesh(center_vertex, radial_coordinates, angular_coordinates, object_mesh):
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
    radial_coordinates = radial_coordinates.copy()
    angular_coordinates = angular_coordinates.copy()

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


def draw_triangles(triangles, points=None, point_color="blue", title="", plot=True):
    """Draws a single triangle and optionally a point in 2D space.

    Parameters
    ----------
    triangles: np.ndarray
        The triangles in cartesian coordinates
    points: np.ndarray
        Points that can optionally also be visualized (in cartesian coordinates)
    point_color: str
        The point color
    title: str
        The title of the plot
    plot: bool
        Whether to plot the image immediately
    """
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    for tri in triangles:
        polygon = Polygon(tri, alpha=.4, edgecolor="red")
        ax.add_patch(polygon)

    if points is not None:
        for point in points:
            ax.scatter(point[0], point[1], color=point_color)

    if points is None:
        ax.set_xlim(triangles[:, :, 0].min(), triangles[:, :, 0].max())
        ax.set_ylim(triangles[:, :, 1].min(), triangles[:, :, 1].max())

    if plot:
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
    contained_gpc_triangles, _ = determine_gpc_triangles(object_mesh, gpc_system)
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


def draw_vertices_in_coordinate_system(radial_coordinates, angular_coordinates):
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
