from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from PIL import Image

import matplotlib
import trimesh
import numpy as np
import io


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


def visualize_lrf(origin, local_reference_frame, shape, scale_lrf=0.05):
    """Visualizes local reference frames.

    Parameters
    ----------
    origin: np.ndarray
        The origin of the local reference frame in form of a 1D-array (vertex).
    local_reference_frame: np.ndarray
        A 2D-array that contains three vectors describing the local reference frame.
    shape: trimesh.Trimesh
        The shape on top of which the local reference frame has been computed.
    scale_lrf: float
        A scaling factor for the local reference frame vectors.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(shape.vertices[:, 0], shape.vertices[:, 1],  shape.vertices[:, 2])
    for vec in local_reference_frame:
        ax.quiver(*origin, *(scale_lrf * vec), color="r")
    plt.show()
