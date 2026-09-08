from geoconv.preprocessing.atlas import Atlas, load_atlas

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import trimesh
import os


def create_grid(n_vertices):
    """Creates an n x n grid of grid vertices.

    Parameters
    ----------
    n_vertices: int
        The number of grid vertices along an edge of the grid.

    Returns
    -------
    trimesh.Trimesh:
        A triangle mesh representing a n x n image, without colors.
    """
    # Get mesh faces
    coordinates = np.linspace(start=0, stop=1, num=n_vertices)
    grid_vertices = np.array([(x, y) for x in coordinates for y in coordinates])
    grid_faces = sp.spatial.Delaunay(grid_vertices).simplices

    # Make vertices 3D (but keep it flat)
    grid_vertices = np.concatenate([grid_vertices, np.zeros(n_vertices ** 2).reshape(-1, 1)], axis=-1)

    return trimesh.Trimesh(vertices=grid_vertices, faces=grid_faces)


def image_to_grid(image, grid):
    """Renders an image on top of a triangular grid.

    Parameters
    ----------
    image: np.ndarray | list
        The image.
    grid: trimesh.Trimesh
        The grid.
    """
    grid_image = trimesh.PointCloud(grid.vertices, colors=plt.cm.binary(np.array(image).reshape((-1))))
    trimesh.Scene([grid, grid_image]).show()


def preprocess_mnist(output_path,
                     max_chart_radius,
                     n_radials,
                     n_angulars,
                     max_temp_radius=None,
                     method="hdm",
                     normalization_method="hdm",
                     processes=1):
    """Preprocesses the MNIST grid atlas and stores barycentric coordinates.

    Parameters
    ----------
    output_path: str
        The output path.
    max_chart_radius: float
        The max chart radius.
    n_radials: list
        A list of possible numbers for radial template coordinates.
    n_angulars: list
        A list of possible numbers for angular template coordinates.
    max_temp_radius: float
        The max temp radius.
    method: str
        The charting algorithm.
    normalization_method: str
        The charting algorithm used to compute the geodesic diameter for mesh normalization.
    processes: int
        The processes.

    Returns
    -------
    Atlas
        The preprocessed atlas.
    """
    output_path = f"{output_path}.hdf5" if not output_path.endswith(".hdf5") else output_path
    if not os.path.isfile(output_path):
        grid = create_grid(n_vertices=28)  # MNIST-images are 28x28 grids
        atlas = Atlas(
            triangle_mesh=grid,
            max_radius=max_chart_radius,
            method=method,
            normalization_method=normalization_method,
            processes=processes
        )
        chart_radii = [
            atlas.min_chart_radius,
            atlas.max_chart_radius,
            atlas.avg_chart_radius,
            atlas.median_chart_radius
        ]
        for n_radial in n_radials:
            for n_angular in n_angulars:
                if max_temp_radius is None:
                    for chart_radius in chart_radii:
                        atlas.determine_barycentric_coordinates(
                            n_radial=n_radial,
                            n_angular=n_angular,
                            template_radius=chart_radius
                        )
                else:
                    atlas.determine_barycentric_coordinates(
                        n_radial=n_radial,
                        n_angular=n_angular,
                        template_radius=max_temp_radius
                    )

        atlas.save(output_path)
        return atlas
    else:
        print(f'Preprocessed dataset already exists at {output_path}.')
        return load_atlas(output_path)
