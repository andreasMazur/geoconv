from geoconv.preprocessing.barycentric_coordinates import create_template_matrix
from geoconv.tensorflow.layers.barycentric_coordinates import compute_bc
from geoconv.tensorflow.utils.compute_shot_lrf import (
    compute_distance_matrix, group_neighborhoods, shot_lrf, logarithmic_map
)
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

    interpolated_template_vertices = []
    for radial_c in range(interpolation_weights.shape[0]):
        for angular_c in range(interpolation_weights.shape[1]):
            interpolated_template_vertex = []
            for idx in range(3):
                closest_neighbor = projections[projections_indices[radial_c, angular_c, idx]]
                interpolation_coefficient = interpolation_weights[radial_c, angular_c, idx]
                interpolated_template_vertex.append(interpolation_coefficient * closest_neighbor)
            interpolated_template_vertex = sum(interpolated_template_vertex)
            interpolated_template_vertices.append(interpolated_template_vertex)
    interpolated_template_vertices = np.array(interpolated_template_vertices).reshape(template.shape)

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


def preprocess_demo(path_to_stanford_bunny, n_radial=5, n_angular=6, n_neighbors=10):
    """Demonstrates and visualizes what the Barycentric-Coordinates layer is doing at the hand of the stanford bunny.

    Download the Stanford bunny from here:
    https://github.com/alecjacobson/common-3d-test-models/blob/master/data/stanford-bunny.zip

    Unzip the .zip-file and move the 'bun_zipper.ply'-file and enter the path to the 'bun_zipper.ply' as argument
    for 'path_to_stanford_bunny'.

    Parameters
    ----------
    path_to_stanford_bunny: str
        The path to the 'bun_zipper.ply'-file containing the stanford bunny.
    n_radial: int
        The amount of radial coordinates for the template in your geodesic convolution.
    n_angular: int
        The amount of angular coordinates for the template in your geodesic convolution.
    n_neighbors: int
        The target amount of vertex-neighbors to have in each neighborhood.
    """
    mpl.use("Qt5Agg")

    # Load the stanford bunny in '.ply'-format as a trimesh-object.
    bunny = load_bunny(path_to_stanford_bunny)
    bunny_vertices = np.array(bunny.vertices).astype(np.float32)

    # Step 1: Calculate the distance matrix
    # shape: (vertices, vertices)
    distance_matrix = compute_distance_matrix(bunny_vertices).numpy()

    # Visualize distance matrix
    visualize_distance_matrix(distance_matrix)

    # Step 2: Retrieve local neighborhood radii (one per neighborhood)
    # shape: (vertices,)
    radii = distance_matrix[np.arange(distance_matrix.shape[0]), np.argsort(distance_matrix, axis=-1)[:, n_neighbors]]

    # Step 3: Determine vertex-neighborhoods
    # 'neighborhoods': (vertices, n_neighbors, 3)
    neighborhoods, neighborhoods_indices = group_neighborhoods(bunny_vertices, radii, n_neighbors, distance_matrix)

    # Visualize three neighborhoods
    for n in np.random.randint(low=0, high=bunny_vertices.shape[0], size=(3,)):
        visualize_neighborhood(bunny_vertices, neighborhoods_indices[n])

    # Step 4: Compute local reference frames
    # 'lrfs': (vertices, 3, 3)
    lrfs = shot_lrf(neighborhoods, radii)

    # Visualize three LRFs
    for lrf_idx in np.random.randint(low=0, high=bunny_vertices.shape[0], size=(3,)):
        visualize_lrf(origin=bunny_vertices[lrf_idx], local_reference_frame=lrfs[lrf_idx], shape=bunny, scale_lrf=0.05)

    # Step 5: Project neighborhoods into the plane spanned by the lrfs using the logarithmic map
    # 'projections': (vertices, n_neighbors, 2)
    projections = logarithmic_map(lrfs, neighborhoods)

    # Visualize three projected neighborhoods
    for n in np.random.randint(low=0, high=bunny_vertices.shape[0], size=(3,)):
        visualize_neighborhood(bunny_vertices, neighborhoods_indices[n])
        visualize_projected_neighborhood(projections[n])

    # Step 6: Configure a template
    # 'template': (n_radial, n_angular, 2)
    template = create_template_matrix(n_radial=n_radial, n_angular=n_angular, radius=radii.mean(), in_cart=True)

    # Step 7: Compute interpolation coefficients for the template vertices within the projections
    # 'interpolation_weights': (vertices, n_radial, n_angular, 3)
    # 'closest_proj': (vertices, n_radial, n_angular, 3)
    # Hereby, 'interpolation_weights[i, j, k, l]' is the BC of neighbor vertex with index 'closest_proj[i, j, k, l]'
    interpolation_weights, closest_proj = compute_bc(template.astype(np.float32), projections)

    # Plot histogram of interpolation weights
    counts, bins = np.histogram(interpolation_weights.numpy().flatten(), bins=100)
    plt.hist(bins[:-1], bins, weights=counts, rwidth=0.5)
    plt.title("Histogram of barycentric coordinates")
    plt.xlabel("barycentric coordinates")
    plt.ylabel("Count")
    plt.show()

    # Visualize three interpolated template vertices in their projected neighborhoods
    for n in np.random.randint(low=0, high=bunny_vertices.shape[0], size=(3,)):
        visualize_interpolations(
            interpolation_weights[n].numpy(), closest_proj[n].numpy(), projections[n].numpy(), template
        )
