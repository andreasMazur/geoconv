from geoconv.preprocessing.barycentric_coordinates import create_template_matrix
from geoconv.tensorflow.layers.barycentric_coordinates import compute_bc
from geoconv.tensorflow.utils.compute_shot_decr import shot_descr
from geoconv.tensorflow.utils.compute_shot_lrf import compute_distance_matrix, logarithmic_map, knn_shot_lrf
from geoconv.utils.visualization import visualize_lrf
from geoconv_examples.stanford_bunny.preprocess_demo import load_bunny

from matplotlib import pyplot as plt

import numpy as np
import trimesh
import matplotlib as mpl


def visualize_interpolations(interpolation_weights, projections_indices, projections, template):
    """Visualizes the interpolated template vertices.

    Parameters
    ----------
    interpolation_weights: np.ndarray
        A 3D-array of shape (n_radial, n_angular, 3) that contains barycentric coordinates (BC).
    projections_indices: np.ndarray
        A 3D-array of shape (n_radial, n_angular, 3) that contains the indices for the BC-
    projections: np.ndarray
        A 2D-array of shape (n_neighbors, 2) that contains the 2D neighborhood projections.
    template: np.ndarray
        A 3D-array of shape (n_radial, n_angular, 2) that contains the 2D template vertices.
    """
    plt.rcParams['text.usetex'] = True

    interpolated_template_vertices = (
            projections[projections_indices] * interpolation_weights.reshape(5, 6, 3, 1)
    ).sum(axis=-2)

    visualize_projected_neighborhood(projections, show=False, color="blue", alpha=1.)
    visualize_projected_neighborhood(template.reshape(-1, 2), show=False, color="red", alpha=.5)
    visualize_projected_neighborhood(interpolated_template_vertices.reshape(-1, 2), show=False, color="green", alpha=.5)

    plt.title("Interpolated template vertices")
    plt.text(0.005, 0.006, r"\textbf{Interpolated}", color="green")
    plt.text(0.005, 0.005, r"\textbf{Not interpolated}", color="red")
    plt.text(0.005, 0.004, r"\textbf{Projected mesh vertices}", color="blue")
    plt.show()


def visualize_projected_neighborhood(projections, show=True, color="blue", alpha=1.):
    """Visualize projected points on the 2D-plane.

    Parameters
    ----------
    projections: np.ndarray
        A 2D-array of shape (n_neighbors, 2) containing 2D-cartesian coordinates.
    show: bool
        Whether to plot immediately.
    color: str
        The color of the projection points.
    alpha: float
        The transparency value of the projection points in between 0 and 1.
    """
    plt.scatter(x=projections[:, 0], y=projections[:, 1], c=color, alpha=alpha)
    plt.grid()
    if show:
        plt.show()


def visualize_neighborhood(mesh_vertices, neighborhood_indices):
    """Visualize a vertex neighborhood on the point cloud.

    Parameters
    ----------
    mesh_vertices: np.ndarray
        A 2D-array of shape (vertices, 3) containing the point cloud vertices
    neighborhood_indices: np.ndarray
        A 1D-array of shape (n_neighbors,) containing the vertex indices for the current neighborhood
    """
    black, red = [0, 0, 0, 255], [255, 0, 0, 255]
    vertex_colors = [black if idx not in neighborhood_indices else red for idx in range(mesh_vertices.shape[0])]
    trimesh.PointCloud(mesh_vertices, colors=vertex_colors).show()


def visualize_distance_matrix(distance_matrix):
    """Visualizes the distance matrix.

    Parameters
    ----------
    distance_matrix: np.ndarray
        A 2D-array that contains the distances between mesh vertices.
    """
    plt.imshow(distance_matrix)
    plt.colorbar()
    plt.title("Euclidean distances between mesh vertices")
    plt.show()


def plot_normals(vertices, lrfs, n_neighbors, take_every_nth=5, axis_limit=0.1, plot=True, length=0.01):
    """Visualize the normals of given LRFs

    Parameters
    ----------
    vertices: np.ndarray
        The vertices that of the point cloud for which the LRFs have been computed.
    lrfs: np.ndarray
        The local reference frames from which to take the normals.
    n_neighbors: int
        The amount of neighbors that have been used to estimate the normals.
    take_every_nth: int
        An integer value that is used to sample a subset from the set of all vertices. Only these will be visualized.
    axis_limit: float
        The axis-limits.
    plot: bool
        Whether to plot immediately.
    length: float
        Length of the normal vectors.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Sample from vertices
    vertices = vertices[::take_every_nth]
    lrfs = lrfs[::take_every_nth]

    # Origins of normal-vectors
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Directions of normal-vectors
    z_axes = lrfs[:, :, 0]
    u, v, w = z_axes[:, 0], z_axes[:, 1], z_axes[:, 2]

    # Plot the vectors at each point in the point cloud
    ax.scatter(x, y, z, color="r")
    ax.quiver(x, y, z, u, v, w, length=length, color="b", alpha=0.33, linewidth=2.5)

    # Plot centroid
    centroid = vertices.mean(axis=-2)
    ax.scatter(centroid[0], centroid[1], centroid[2], color="g")

    # Set axis limits
    ax.set_xlim([-axis_limit, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])
    ax.set_zlim([-axis_limit, axis_limit])

    plt.title(f"Normals calculated using {n_neighbors} neighbors.")

    # Show the plot
    if plot:
        plt.show()


def preprocess_demo(path_to_stanford_bunny,
                    n_neighbors,
                    template_radius=None,
                    n_radial=5,
                    n_angular=6,
                    visualize_dist_matrix=True,
                    visualize_lrf_normals=True,
                    visualize_shot_descr=True,
                    visualize_neighborhoods=True,
                    visualize_lrfs=True,
                    visualize_projections=True,
                    visualize_bc_histogram=True,
                    visualize_bc_interpolations=True):
    """Demonstrates and visualizes what the Barycentric-Coordinates layer is doing at the hand of the stanford bunny.

    Download the Stanford bunny from here:
    https://github.com/alecjacobson/common-3d-test-models/blob/master/data/stanford-bunny.zip

    Unzip the .zip-file and move the 'bun_zipper.ply'-file and enter the path to the 'bun_zipper.ply' as argument
    for 'path_to_stanford_bunny'.

    Parameters
    ----------
    path_to_stanford_bunny: str
        The path to the 'bun_zipper.ply'-file containing the stanford bunny.
    n_neighbors: int
        The amount of vertices to use during tangent plane estimation
    template_radius: float | None
        The radius of the template to train.
    n_radial: int
        The amount of radial coordinates for the template in your geodesic convolution.
    n_angular: int
        The amount of angular coordinates for the template in your geodesic convolution.
    visualize_dist_matrix: bool
        Whether to visualize the distance matrix.
    visualize_lrf_normals: bool
        Whether to visualize the normal vectors given by local reference frames (LRFs).
    visualize_shot_descr: bool
        Whether to visualize SHOT-descriptors.
    visualize_neighborhoods: bool
        Whether to visualize local neighborhoods on the 3D point-cloud.
    visualize_lrfs: bool
        Whether to visualize entire local reference frames (LRFs).
    visualize_projections: bool
        Whether to visualize neighborhoods that are projected on tangent planes given by local reference frames (LRFs).
    visualize_bc_histogram: bool
        Whether to visualize a histogram that shows the distribution of the computed barycentric coordinates (BC).
    visualize_bc_interpolations: bool
        Whether to visualize interpolated template vertices.
    """
    mpl.use("Qt5Agg")

    # Load the stanford bunny in '.ply'-format as a trimesh-object.
    bunny = load_bunny(path_to_stanford_bunny)
    bunny_vertices = np.array(bunny.vertices).astype(np.float32)
    bunny_vertices = bunny_vertices - bunny_vertices.mean(axis=-2)

    # Step 1: Calculate the distance matrix
    # shape: (vertices, vertices)
    distance_matrix = compute_distance_matrix(bunny_vertices).numpy()

    # Visualize distance matrix
    if visualize_dist_matrix:
        visualize_distance_matrix(distance_matrix)

    # Compute local reference frames (LRFs)
    if visualize_lrf_normals:
        for n in [5, 15, 25, 35, 45]:
            # Visualize lrf normals (z-axes)
            lrfs, neighborhoods, neighborhoods_indices = knn_shot_lrf(n, bunny_vertices)
            lrfs = lrfs.numpy()
            plot_normals(bunny_vertices, lrfs, n, plot=False)
        plt.show()
    lrfs, neighborhoods, neighborhoods_indices = knn_shot_lrf(n_neighbors, bunny_vertices)
    lrfs = lrfs.numpy()

    # Compute SHOT descriptor
    shot_descriptor = shot_descr(
        neighborhoods=neighborhoods,
        normals=lrfs[:, :, 0],
        neighborhood_indices=neighborhoods_indices,
        radius=np.max(neighborhoods)
    )

    if visualize_shot_descr:
        plt.imshow(shot_descriptor[::5])
        plt.show()

    # Visualize three neighborhoods
    if visualize_neighborhoods:
        for n in np.random.randint(low=0, high=bunny_vertices.shape[0], size=(3,)):
            visualize_neighborhood(bunny_vertices, neighborhoods_indices[n])

    # Visualize three LRFs
    if visualize_lrfs:
        for lrf_idx in np.random.randint(low=0, high=bunny_vertices.shape[0], size=(3,)):
            visualize_lrf(
                origin=bunny_vertices[lrf_idx],
                local_reference_frame=lrfs[lrf_idx],
                vertices=bunny_vertices,
                scale_lrf=0.05
            )

    # Step 5: Project neighborhoods into the plane spanned by the lrfs using the logarithmic map
    # 'projections': (vertices, n_neighbors, 2)
    projections = logarithmic_map(lrfs, neighborhoods)

    # Visualize three projected neighborhoods
    if visualize_projections:
        for n in np.random.randint(low=0, high=bunny_vertices.shape[0], size=(3,)):
            visualize_neighborhood(bunny_vertices, neighborhoods_indices[n])
            visualize_projected_neighborhood(projections[n])

    # Step 6: Configure a template
    if template_radius is None:
        # If no template radius is given, select the average Euclidean distance to the n-th neighbor
        template_radius = distance_matrix[
            np.arange(distance_matrix.shape[0]), np.argsort(distance_matrix, axis=-1)[:, n_neighbors]
        ]
        template_radius = np.mean(template_radius)
    # 'template': (n_radial, n_angular, 2)
    template = create_template_matrix(n_radial=n_radial, n_angular=n_angular, radius=template_radius, in_cart=True)

    # Step 7: Compute interpolation coefficients for the template vertices within the projections
    # 'interpolation_weights': (vertices, n_radial, n_angular, 3)
    # 'closest_proj': (vertices, n_radial, n_angular, 3)
    # Hereby, 'interpolation_weights[i, j, k, l]' is the BC of neighbor vertex with index 'closest_proj[i, j, k, l]'
    interpolation_weights, closest_proj = compute_bc(template.astype(np.float32), projections)

    # Plot histogram of interpolation weights
    if visualize_bc_histogram:
        counts, bins = np.histogram(interpolation_weights.numpy().flatten(), bins=100)
        plt.hist(bins[:-1], bins, weights=counts, rwidth=0.5)
        plt.title("Histogram of barycentric coordinates")
        plt.xlabel("barycentric coordinates")
        plt.ylabel("Count")
        plt.show()

    # Visualize three interpolated template vertices in their projected neighborhoods
    if visualize_bc_interpolations:
        for n in np.random.randint(low=0, high=bunny_vertices.shape[0], size=(3,)):
            visualize_interpolations(
                interpolation_weights[n].numpy(), closest_proj[n].numpy(), projections[n].numpy(), template
            )
