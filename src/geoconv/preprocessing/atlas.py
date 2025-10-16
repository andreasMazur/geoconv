from geoconv.preprocessing.distance_computation import normalize_shape, calculate_local_charts

from tqdm import tqdm

import numpy as np
import trimesh
import h5py


def determine_faces_for_charts(triangle_mesh, local_charts):
    """For each local chart, this function extracts those faces for which local coordinates exist.

    Parameters
    ----------
    triangle_mesh : trimesh.Trimesh
        The underlying triangle mesh for the given local charts.
    local_charts : np.ndarray
        The local charts for the given triangle mesh.

    Returns
    -------
    dict:
        The completely included triangles within the local charts.
    """
    available_faces = {}
    for chart_idx, chart in tqdm(enumerate(local_charts), desc="Extracting faces from local charts"):
        faces_in_local_coords = chart[triangle_mesh.faces]
        mask = faces_in_local_coords[..., 0] != np.inf
        all_coords_available = mask.astype(np.int32).prod(axis=-1).astype(np.bool_)
        available_faces[chart_idx] = np.array(triangle_mesh.faces[all_coords_available])
    return available_faces


def load_atlas(filepath):
    with h5py.File(filepath, "r") as f:
        # Load mesh
        vertices = np.array(f["triangle_mesh/vertices"])
        faces = np.array(f["triangle_mesh/faces"])
        triangle_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Load charts information
        charts = np.array(f[f"charts_information/charts"])

        # Load chart face information
        chart_faces = {}
        for origin_vertex_idx in range(vertices.shape[0]):
            chart_faces[origin_vertex_idx] = np.array(f[f"charts_faces/{origin_vertex_idx}"])

        # Load chart triangle information
        chart_triangles = {}
        for origin_vertex_idx in range(vertices.shape[0]):
            chart_triangles[origin_vertex_idx] = np.array(f[f"charts_triangles/{origin_vertex_idx}"])

        # Load meta information
        max_radius = f.attrs.get("max_radius")
        method = f.attrs.get("method")
        processes = f.attrs.get("processes")

    # Instantiate loaded atlas
    atlas = Atlas.__new__(Atlas)

    # Meta information
    atlas.max_radius = max_radius
    atlas.method = method
    atlas.processes = processes

    # Triangle mesh and charts information
    atlas.triangle_mesh = triangle_mesh
    atlas.charts = charts
    atlas.chart_faces = chart_faces
    atlas.chart_triangles = chart_triangles

    # Return instantiated atlas
    return atlas


class Atlas:
    def __init__(self, triangle_mesh, max_radius, method="hdm", processes=1):
        # Meta information
        self.max_radius = max_radius
        self.method = method
        self.processes = processes

        # Triangle mesh and atlas information
        self.triangle_mesh = normalize_shape(triangle_mesh, method=method, processes=processes)
        self.charts = calculate_local_charts(
            triangle_mesh,
            method=method,
            processes=processes,
            max_radius=max_radius,
            calculate_angle=True
        )
        self.chart_faces = determine_faces_for_charts(triangle_mesh, self.charts)
        self.chart_triangles = {k: np.array(self.charts[k][v]) for k, v in self.chart_faces.items()}

    def save(self, filepath):
        with h5py.File(filepath, "w") as f:
            # Save mesh information
            h5_triangle_mesh = f.create_group("triangle_mesh")
            h5_triangle_mesh.create_dataset(
                "vertices", data=np.asarray(self.triangle_mesh.vertices), compression="gzip"
            )
            h5_triangle_mesh.create_dataset("faces", data=np.asarray(self.triangle_mesh.faces), compression="gzip")

            # Save chart information
            h5_charts_information = f.create_group("charts_information")
            h5_charts_information.create_dataset("charts", data=self.charts)

            # Save chart face information
            h5_charts_faces_information = f.create_group("charts_faces")
            for k, v in self.chart_faces.items():
                h5_charts_faces_information.create_dataset(f"{k}", data=v, compression="gzip")

            # Save chart triangle information
            h5_charts_triangle_information = f.create_group("charts_triangles")
            for k, v in self.chart_triangles.items():
                h5_charts_triangle_information.create_dataset(f"{k}", data=v, compression="gzip")

            # Save meta information
            f.attrs["max_radius"] = self.max_radius
            f.attrs["method"] = self.method
            f.attrs["processes"] = self.processes
