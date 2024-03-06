from geoconv.preprocessing.barycentric_coordinates import polar_to_cart

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from PIL import Image

import os
import matplotlib
import trimesh
import numpy as np
import matplotlib.cm as cm
import io
import time


def draw_barycentric_coordinates(gpc_system, barycentric_coordinates, save_name=""):
    """Draws barycentric coordinates successively.

    Parameters
    ----------
    gpc_system: GPCSystem
        The GPC-system in which the barycentric coordinates should be illustrated.
    barycentric_coordinates: np.ndarray
        The barycentric coordinates to illustrate.
    save_name: str
        If given, the plot will be saved under this path.
    """
    interpolated_points = []
    for rc in range(barycentric_coordinates.shape[0]):
        for ac in range(barycentric_coordinates.shape[1]):
            triangle_indices = barycentric_coordinates[rc, ac, :, 0].astype(np.int16)
            triangle_interpolation_coefficients = barycentric_coordinates[rc, ac, :, 1]
            triangle_coordinates = np.stack(
                [gpc_system.x_coordinates[triangle_indices], gpc_system.y_coordinates[triangle_indices]],
                axis=-1
            )
            dot_interpolation = triangle_coordinates.T @ triangle_interpolation_coefficients
            interpolated_points.append(dot_interpolation)
            pc = ["blue" for _ in range(len(interpolated_points) - 1)] + ["yellow"] + ["orange" for _ in range(3)]
            draw_triangles(
                gpc_system.get_gpc_triangles(in_cart=True),
                points=np.array(interpolated_points + list(triangle_coordinates)),
                point_color=pc,
                title=f"Interpolation Coefficients: {triangle_interpolation_coefficients}",
                plot=True,
                save_name=f"{save_name}_{rc}_{ac}"
            )


def draw_multiple_princeton_benchmarks(save_name, **kwargs):
    """Draws the Princeton benchmarks all given numpy files.

    Parameters
    ----------
    save_name: str
        The name of the file in which the result plot will be saved.
    kwargs: str
        The keys will be displayed as a title. The values are the paths to the numpy files, the line-style
        and the color.
    """

    for name, (path, line_style, color) in kwargs.items():
        # Load values from princeton benchmark
        pb_values = np.load(path)

        # Filter values
        unique_x_values = np.unique(pb_values[:, 1])
        unique_values = []
        for unique_x in unique_x_values:
            unique_values.append(pb_values[np.where(pb_values[:, 1] == unique_x)[0][-1]])
        unique_values = np.array(unique_values)

        # Plot values
        plt.plot(unique_values[:, 1], unique_values[:, 0], linestyle=line_style, label=name, c=color)
        # plt.title(f"Princeton Benchmarks")
        plt.xlabel("geodesic error")
        plt.ylabel("\% correct correspondences")

    plt.grid()
    plt.legend()
    plt.savefig(f"{save_name}.svg")
    plt.show()


def draw_prior(icnn_layer, indices):
    """Wrapper method for 'draw_interpolation_coefficients_single_idx'

    Parameters
    ----------
    icnn_layer: src.layers.conv_intrinsic.ConvIntrinsic
        The for which the interpolation coefficients shall be visualized
    indices: List[int]
        A list of index-tuple for accessing template vertices. I.e. indices[x] = (a, b) with K[a, b] = (rho, theta).
    """
    fig = plt.figure(figsize=(10, 5))
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
    """Visualizes the interpolation coefficients of the patch operator at a specific template vertex

    Parameters
    ----------
    icnn_layer: src.layers.conv_intrinsic.ConvIntrinsic
        The for which the interpolation coefficients shall be visualized
    radial_idx: int
        The index of the radial coordinate from the template vertex for which we visualize the interpolation
        coefficients
    angular_idx: int
        The index of the angular coordinate from the template vertex for which we visualize the interpolation
        coefficients
    fig:
        The figure in which to plot the axes
    axis_ic:
        The axis on which to plot the interpolation coefficients matrix
    axis_kv:
        The axis on which to plot the weighted template vertices
    """

    # Get interpolation coefficients of the given layer: I[a, b] \in R^{n_radial * n_angular}
    weights = icnn_layer._kernel[radial_idx, angular_idx].numpy()
    template_size = icnn_layer._template_size
    template_matrix = icnn_layer._template_vertices.numpy()

    # Reshape vector into matrix: (n_radial * n_angular,) -> (n_radial, n_angular)
    # See 'ConvIntrinsic._configure_patch_operator()' for why it is stored as a vector.
    weights = weights.reshape(template_size)

    # Visualize interpolation coefficient matrix
    pos = axis_ic.matshow(weights, cmap="rainbow")
    fig.colorbar(pos, ax=axis_ic, fraction=0.046, pad=0.04)
    axis_ic.set_title(f"Prior as Matrix")
    template_matrix = template_matrix.reshape((-1, 2))

    # TODO: For some reason, matplotlib only shows all labels from list[1:] and forgets about list[0]
    axis_ic.set_ylabel("Radial coordinate")
    axis_ic.set_yticklabels(["0"] + [f"{x:.3f}" for x in np.unique(template_matrix[:, 0])])

    axis_ic.set_xlabel("Angular coordinate (in radian)")
    axis_ic.set_xticklabels(["0"] + [f"{x:.3f}" for x in np.unique(template_matrix[:, 1])])

    # Visualize interpolation coefficients at template vertices
    axis_kv.scatter(
        template_matrix[:, 1], template_matrix[:, 0], c=cm.rainbow(weights.flatten()), s=150, edgecolor="black"
    )
    axis_kv.set_title("Prior on Template Vertices")
    axis_kv.grid(True)
    axis_kv.set_axisbelow(True)


def draw_correspondences(query_mesh, prediction, reference_mesh, color_map="Reds", save_image=True):
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
    save_image: bool
        Whether to save the image
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

    scene = trimesh.Scene([query_mesh, query_mesh_pc, reference_mesh, reference_mesh_pc])
    scene.show()
    if save_image:
        image_bytes = scene.save_image(resolution=(1080, 1080))
        image_array = np.array(Image.open(io.BytesIO(image_bytes)))
        matplotlib.image.imsave("./correspondence.png", image_array)


def draw_gpc_on_mesh(center_vertex,
                     radial_coordinates,
                     angular_coordinates,
                     object_mesh,
                     save_name="",
                     angles=(0., 0., 0.),
                     distance=1.,
                     center=(0., 0., 0.)):
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
    save_name: str
        The name of the image. If none is given, the image will not be saved.
    angles: tuple
        A 3-tuple that describes the camera angle in radians.
    distance: float
        A float that describes the camera distance.
    center: tuple
        A 3-tuple that describes the camera center.
    """
    radial_coordinates = radial_coordinates.copy()
    angular_coordinates = angular_coordinates.copy()

    object_mesh.visual.vertex_colors = [90, 90, 90, 200]

    # Visualize radial coordinates
    radial_coordinates[radial_coordinates == np.inf] = 0.0
    colors = trimesh.visual.interpolate(radial_coordinates, color_map="Reds")
    colors[center_vertex] = np.array([255, 255, 0, 255])
    point_cloud_1 = trimesh.points.PointCloud(object_mesh.vertices, colors=colors)
    to_visualize = [object_mesh, point_cloud_1]
    scene = trimesh.Scene(to_visualize)
    scene.set_camera(angles=angles, distance=distance, center=center)
    scene.show()
    if save_name:
        image_bytes = scene.save_image(resolution=(3840, 3840))
        image_array = np.array(Image.open(io.BytesIO(image_bytes)))
        matplotlib.image.imsave(f"{save_name}_radial_coords.png", image_array)

    # Visualize angular coordinates
    colors = trimesh.visual.interpolate(angular_coordinates, color_map="YlGn")
    colors[center_vertex] = np.array([255, 255, 0, 255])
    point_cloud_2 = trimesh.points.PointCloud(object_mesh.vertices, colors=colors)
    to_visualize.append(point_cloud_2)
    scene = trimesh.Scene(to_visualize)
    scene.set_camera(angles=angles, distance=distance, center=center)
    scene.show()
    if save_name:
        image_bytes = scene.save_image(resolution=(3840, 3840))
        image_array = np.array(Image.open(io.BytesIO(image_bytes)))
        matplotlib.image.imsave(f"{save_name}_angular_coords.png", image_array)


def draw_triangles(triangles, points=None, point_color="blue", title="", plot=True, save_name=""):
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
    save_name: str
        If given, the plot will be stored under this path.
    """
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    for tri in triangles:
        polygon = Polygon(tri, alpha=.4, edgecolor="red")
        ax.add_patch(polygon)

    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], color=point_color)

    if points is None:
        ax.set_xlim(triangles[:, :, 0].min(), triangles[:, :, 0].max())
        ax.set_ylim(triangles[:, :, 1].min(), triangles[:, :, 1].max())

    plt.grid()
    if save_name:
        plt.savefig(f"{save_name}.svg")

    if plot:
        plt.show()


def draw_gpc_triangles(gpc_system,
                       template_matrix=None,
                       alpha=.4,
                       edge_color="red",
                       scatter_color="green",
                       highlight_face=-1,
                       plot=True,
                       title="",
                       save_name=""):
    """Draws the triangles of a local GPC-system.

    Parameters
    ----------
    gpc_system: GPCSystem
        The GPC-system to visualize.
    template_matrix: np.ndarray
        A 3D-array that describes template vertices in cartesian coordinates. If 'None' is passed
        no template vertices will be visualized.
    alpha: float
        The opacity of the polygons
    edge_color: str
        The color for the triangle edges
    scatter_color: str
        The color for the template vertices (in case a template is given)
    highlight_face: int
        The index of a triangle, which shall be highlighted
    plot: bool
        Whether to immediately plot
    title: str
        The title of the plot
    save_name: str
        The name of the image. If none is given, the image will not be saved.
    """
    gpc_system_faces = gpc_system.get_gpc_triangles(in_cart=True)

    min_coordinate = gpc_system_faces.min()
    max_coordinate = gpc_system_faces.max()

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlim([min_coordinate, max_coordinate])
    ax.set_ylim([min_coordinate, max_coordinate])
    polygons = PolyCollection(gpc_system_faces, alpha=alpha, edgecolors=edge_color)
    ax.add_collection(polygons)

    # Print template
    if template_matrix is not None:
        for radial_idx in range(template_matrix.shape[0]):
            ax.scatter(template_matrix[radial_idx, :, 0], template_matrix[radial_idx, :, 1], color=scatter_color)

    # Highlight triangle
    if highlight_face > -1:
        ax.add_patch(
            Polygon(gpc_system_faces[highlight_face], linewidth=3., fill=False, edgecolor="purple")
        )
        ax.scatter(
            gpc_system_faces[highlight_face][:, 0],
            gpc_system_faces[highlight_face][:, 1],
            s=90.,
            color="purple"
        )
        for idx, annotation in enumerate(["a", "b", "c"]):
            x = gpc_system_faces[highlight_face][idx, 0]
            y = gpc_system_faces[highlight_face][idx, 1]
            ax.annotate(annotation, (x, y), fontsize=15)

    if save_name:
        plt.savefig(save_name)

    if plot:
        plt.show()

    plt.close(fig)
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


def draw_edge_cache(edge_cache,
                    u,
                    theta,
                    edges_to_highlight=None,
                    point_to_highlight=None,
                    highlighting_color="red",
                    saving_folder="./visualization"):
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    fig, ax = plt.subplots()
    for edge in edge_cache[-1]:
        vertex_1 = polar_to_cart(angles=theta[edge[0]], scales=u[edge[0]])
        vertex_2 = polar_to_cart(angles=theta[edge[1]], scales=u[edge[1]])
        ax.plot([vertex_1[0], vertex_2[0]], [vertex_1[1], vertex_2[1]], c="blue")
        ax.annotate(f"{edge[0]}", vertex_1)
        ax.annotate(f"{edge[1]}", vertex_2)
    if edges_to_highlight is not None:
        for edge in edges_to_highlight:
            vertex_1 = polar_to_cart(angles=theta[edge[0]], scales=u[edge[0]])
            vertex_2 = polar_to_cart(angles=theta[edge[1]], scales=u[edge[1]])
            ax.plot([vertex_1[0], vertex_2[0]], [vertex_1[1], vertex_2[1]], c=highlighting_color)
            ax.scatter([vertex_1[0], vertex_2[0]], [vertex_1[1], vertex_2[1]], c=highlighting_color)
            ax.annotate(f"{edge[0]}", vertex_1)
            ax.annotate(f"{edge[1]}", vertex_2)
    if point_to_highlight is not None:
        x, y = polar_to_cart(angles=point_to_highlight[2], scales=point_to_highlight[1])
        ax.scatter([x], [y], c="green")
        ax.annotate(point_to_highlight[0], [x, y])
    plt.savefig(f"./{saving_folder}/{time.time()}.svg")
    plt.close()
    time.sleep(.5)
