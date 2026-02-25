from geoconv.preprocessing.dgpc.wrapper import pickable_dgpc
from geoconv.preprocessing.fmm.wrapper import pickable_fmm
from geoconv.preprocessing.hdm.wrapper import pickable_hdm
from geoconv.preprocessing.tp.wrapper import pickable_tp
from geoconv.preprocessing.tp.angles import compute_angles_for_distances

from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import math
import trimesh


def calculate_local_charts(triangle_mesh,
                           method="hdm",
                           processes=1,
                           max_radius=np.inf,
                           calculate_angle=True,
                           process_description=""):
    """Calculates local charts on triangle meshes.

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
    calculate_angle: bool
        Whether to calculate angles.
    process_description: str
        A description to show in the progress bar.

    Returns
    -------
    np.ndarray:
        The computed local charts.
    """
    assert processes > 0, "Number of processes must be greater than 0."
    assert method in ["hdm", "fmm", "dgpc", "tp"], (
        "Choose either 'hdm', 'fmm', 'dgpc' or 'tp' as distance calculation method."
    )

    mesh_vertices = np.array(triangle_mesh.vertices)
    mesh_faces = np.array(triangle_mesh.faces)
    n_vertices = mesh_vertices.shape[0]

    # Divide indices into subsets for which the solver should calculate distances in parallel
    all_vertex_indices = np.arange(n_vertices)
    if n_vertices % processes != 0:
        chunk_size = math.floor(n_vertices / processes)
        index_subsets = [all_vertex_indices[p * chunk_size:(p + 1) * chunk_size] for p in range(processes+1)]
    else:
        index_subsets = np.split(all_vertex_indices, processes)

    # Compute geodesic distances (and angles if method = "dgpc")
    with Pool(processes) as p:
        # Choose method
        if method == "hdm":
            method_fn = pickable_hdm
            arg_list = [(idx_subset, mesh_vertices, mesh_faces) for idx_subset in index_subsets]
        elif method == "fmm":
            method_fn = pickable_fmm
            arg_list = [(idx_subset, mesh_vertices, mesh_faces) for idx_subset in index_subsets]
        elif method == "dgpc":
            method_fn = pickable_dgpc
            arg_list = [(idx_subset, mesh_vertices, mesh_faces, max_radius) for idx_subset in index_subsets]
        elif method == "tp":
            method_fn = pickable_tp
            arg_list = [(idx_subset, mesh_vertices, max_radius) for idx_subset in index_subsets]
        else:
            raise RuntimeError("Unknown method for calculating geodesic distances.")

        # Initiate parallel computation
        if process_description == "":
            process_description = f"Computing local charts using '{method}'"
        distances = p.starmap(
            method_fn,
            tqdm(arg_list, total=len(index_subsets), postfix=process_description)
        )
    distances = np.concatenate(distances, axis=0)

    if method in ["dgpc", "tp"]:
        # The DGPC/TP-algorithm returns both radial- and angular coordinates.
        # Thus, if user expects only distances, we extract them here.
        if not calculate_angle:
            distances = distances[..., 0]

        # DGPC algorithm computes distances only up to max_radius internally.
        return distances
    distances[distances > max_radius] = np.inf

    if calculate_angle:
        # Compute angles using tangent plane projections
        angles = compute_angles_for_distances(triangle_mesh, distances)

        # Combine distances and angles to local charts
        return np.stack([distances, angles], axis=-1)
    else:
        return distances


def normalize_shape(triangle_mesh, method="hdm", processes=1):
    """Normalizes mesh by scaling its geodesic diameter to one and moving its point of mass to zero.

    Parameters
    ----------
    triangle_mesh: trimesh.Trimesh
        The mesh to normalize.
    method: str
        Either "hdm" (heat diffusion for distance calc.) or "fmm" (fast marching method). Defaults to "hdm".
    processes: int
        The number of processes to use for parallel computation.

    Returns
    -------
    trimesh.Trimesh:
        The normalized triangle mesh.
    """
    assert method in ["hdm", "fmm"], "For normalization, choose either 'hdm' or 'fmm' as distance calculation method."

    distances = calculate_local_charts(
        triangle_mesh,
        method=method,
        processes=processes,
        max_radius=np.inf,
        calculate_angle=False,
        process_description=f"Normalizing shape using {method}"
    )
    geodesic_diameter = distances.max()
    normalized_vertices = triangle_mesh.vertices / geodesic_diameter
    normalized_vertices = normalized_vertices - np.mean(normalized_vertices, axis=0)

    return trimesh.Trimesh(vertices=normalized_vertices, faces=np.array(triangle_mesh.faces)), geodesic_diameter

