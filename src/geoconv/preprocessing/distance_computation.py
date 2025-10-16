from geoconv.preprocessing.angle_computation import compute_angles
from geoconv.preprocessing.fmm.wrapper import pickable_fmm
from geoconv.preprocessing.hdm.wrapper import pickable_hdm

from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import math
import trimesh


def calc_local_charts_wrapper(triangle_mesh, method="hdm", processes=1, max_radius=np.inf, normalize_shape=True):
    """Wrapper for the Heat Diffusion Method for Distance Calculations from Potpourri3D.

    The heat diffusion for distance calculation has been published in:
    > [The heat method for distance computation](https://dl.acm.org/doi/10.1145/3131280)
    > Keenan Crane, Clarisse Weischedel, Max Wardetzky

    Code available on GitHub:
    > https://github.com/nmwsharp/potpourri3d

    Parameters
    ----------
    triangle_mesh: trimesh.Trimesh
        The triangle mesh to compute distances on.
    method: str
        Either "hdm" (heat diffusion for distance calc.) or "fmm" (fast marching method). Defaults to "hdm".
    processes: int
        The number of processes to use for parallel computation.
    max_radius: float
        The maximum radius for a local chart.
    normalize_shape: bool
        Whether to normalize the shape by its geodesic diameter before computing the local charts.

    Returns
    -------
    np.ndarray:
        The computed local charts.
    """
    assert processes > 0, "Number of processes must be greater than 0."
    assert method in ["hdm", "fmm"], "Choose either 'hdm' or 'fmm' as distance calculation method."

    mesh_vertices = np.array(triangle_mesh.vertices)
    mesh_faces = np.array(triangle_mesh.faces)
    n_vertices = mesh_vertices.shape[0]

    # Divide indices into subsets for which the solver should calculate distances in parallel
    chunk_size = math.floor(n_vertices / processes)
    all_vertex_indices = np.arange(n_vertices)
    if n_vertices % processes != 0:
        index_subsets = [all_vertex_indices[p * chunk_size:(p + 1) * chunk_size] for p in range(processes+1)]
    else:
        index_subsets = np.split(all_vertex_indices, processes)

    # Compute geodesic distances
    with Pool(processes) as p:
        distances = p.starmap(
            pickable_hdm if method == "hdm" else pickable_fmm,
            tqdm(
                [(idx_subset, mesh_vertices, mesh_faces) for idx_subset in index_subsets],
                total=len(index_subsets),
                postfix=f"Computing local charts using '{method}'",
            )
        )
    distances = np.concatenate(distances, axis=0)
    distances[distances > max_radius] = np.inf

    # Compute angles using tangent plane projections
    angles = compute_angles(triangle_mesh, distances)

    # Combine distances and angles to local charts
    return np.stack([distances, angles], axis=-1)


if __name__ == "__main__":
    mesh = trimesh.load_mesh("/home/andreas/Uni/datasets/MPI-FAUST/training/registrations/tr_reg_000.ply")
    calc_local_charts_wrapper(mesh, method="fmm", max_radius=0.05, processes=10, normalize_shape=True)
    calc_local_charts_wrapper(mesh, method="hdm", max_radius=0.05, processes=10, normalize_shape=True)
