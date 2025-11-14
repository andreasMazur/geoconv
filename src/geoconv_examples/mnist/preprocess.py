from geoconv.preprocessing.atlas import Atlas, load_atlas

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import trimesh
import os


def create_grid(n_vertices):
    # Get mesh faces
    coordinates = np.linspace(start=0, stop=1, num=n_vertices)
    grid_vertices = np.array([(x, y) for x in coordinates for y in coordinates])
    grid_faces = sp.spatial.Delaunay(grid_vertices).simplices

    # Make vertices 3D (but keep it flat)
    grid_vertices = np.concatenate([grid_vertices, np.zeros(n_vertices ** 2).reshape(-1, 1)], axis=-1)

    return trimesh.Trimesh(vertices=grid_vertices, faces=grid_faces)


def image_to_grid(image, grid):
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
                            radius=chart_radius
                        )
                else:
                    atlas.determine_barycentric_coordinates(
                        n_radial=n_radial,
                        n_angular=n_angular,
                        radius=max_temp_radius
                    )

        atlas.save(output_path)
        return atlas
    else:
        print(f'Preprocessed dataset already exists at {output_path}.')
        return load_atlas(output_path)
