from tqdm import tqdm

import sys
import numpy as np
import pygeodesic.geodesic as geodesic


def safe_calculate_distance_matrix(mesh_path):
    """Compute the distance matrix over all vertices of a given mesh and store in those in a *.npy-file.

    Execute pygeodesic in a standalone program to catch errors of the pygeodesic-package.

    Parameters
    ----------
    mesh_path: str
        The path from where to load the triangle mesh.
    """
    vertices = np.load(f"{mesh_path}/mesh_vertices.npy")
    faces = np.load(f"{mesh_path}/mesh_faces.npy")

    n_vertices = vertices.shape[0]
    distance_matrix = np.zeros((n_vertices, n_vertices))

    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
    for sp in tqdm(range(n_vertices), postfix=f"Calculating geodesic diameter.."):
        distances, _ = geoalg.geodesicDistances([sp], None)
        distance_matrix[sp] = distances

    np.save(f"{mesh_path}/distance_matrix.npy", distance_matrix)


if __name__ == "__main__":
    argc, argv = sys.argv
    safe_calculate_distance_matrix(argv)
