from geoconv.preprocessing.bc.bc_utils import polar_to_cart
from geoconv.preprocessing.bc.wrapper import compute_barycentric_coordinates
from geoconv.preprocessing.distance_computation import normalize_shape, calculate_local_charts

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle

import matplotlib.cm as cm
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
        chart_radii = np.array(f[f"charts_information/chart_radii"])

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
        original_geodesic_diameter = f.attrs.get("original_geodesic_diameter")

        # Load chart statistics
        max_chart_radius = f.attrs.get("max_chart_radius")
        min_chart_radius = f.attrs.get("min_chart_radius")
        avg_chart_radius = f.attrs.get("avg_chart_radius")
        std_chart_radius = f.attrs.get("std_chart_radius")
        median_chart_radius = f.attrs.get("median_chart_radius")

    # Instantiate loaded atlas
    atlas = Atlas.__new__(Atlas)

    # Set meta information
    atlas.max_radius = max_radius
    atlas.method = method
    atlas.processes = processes
    atlas.original_geodesic_diameter = original_geodesic_diameter

    # Set chart statistics
    atlas.max_chart_radius = max_chart_radius
    atlas.min_chart_radius = min_chart_radius
    atlas.avg_chart_radius = avg_chart_radius
    atlas.std_chart_radius = std_chart_radius
    atlas.median_chart_radius = median_chart_radius

    # Set triangle mesh and charts
    atlas.triangle_mesh = triangle_mesh
    atlas.charts = charts
    atlas.charts_radii = chart_radii
    atlas.chart_faces = chart_faces
    atlas.chart_triangles = chart_triangles

    # Return instantiated atlas
    return atlas


class Atlas:
    def __init__(self, triangle_mesh, max_radius, method="hdm", normalization_method="hdm", processes=1):
        # Meta information
        self.max_radius = max_radius
        self.method = method
        self.processes = processes

        # Normalize mesh
        self.triangle_mesh, geodesic_diameter = normalize_shape(
            triangle_mesh,
            method=normalization_method,
            processes=processes
        )
        self.original_geodesic_diameter = geodesic_diameter

        # Local charts
        self.charts = calculate_local_charts(
            triangle_mesh,
            method=method,  # DGPC does not work well for normalization
            processes=processes,
            max_radius=max_radius,
            calculate_angle=True
        )

        # Compute statistics about chart radii
        self.chart_radii = np.array([distances[distances != np.inf].max() for distances in self.charts[..., 0]])
        self.max_chart_radius = self.chart_radii.max()
        self.min_chart_radius = self.chart_radii.min()
        self.avg_chart_radius = self.chart_radii.mean()
        self.std_chart_radius = self.chart_radii.std()
        self.median_chart_radius = np.median(self.chart_radii)

        # Translate charts into cartesian coordinates (required by BC-computation)
        self.charts = polar_to_cart(self.charts[..., 1], self.charts[..., 0])

        # Store faces and triangles
        self.chart_faces = determine_faces_for_charts(triangle_mesh, self.charts)
        self.chart_triangles = {k: np.array(self.charts[k][v]) for k, v in self.chart_faces.items()}

        # Placeholder attribute for barycentric coordinates
        self.barycentric_coordinates = {}

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
            h5_charts_information.create_dataset("chart_radii", data=self.chart_radii)

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
            f.attrs["original_geodesic_diameter"] = self.original_geodesic_diameter

            # Save chart radius statistics
            f.attrs["max_chart_radius"] = self.max_chart_radius
            f.attrs["min_chart_radius"] = self.min_chart_radius
            f.attrs["avg_chart_radius"] = self.avg_chart_radius
            f.attrs["std_chart_radius"] = self.std_chart_radius
            f.attrs["median_chart_radius"] = self.median_chart_radius

    def visualize_chart(self, chart_idx, visualize_3d=False, show_statistics=True):
        if visualize_3d:
            # Cartesian to polar conversion for visualization
            selected_chart = self.charts[chart_idx][self.charts[chart_idx, :, 0] != np.inf]
            selected_chart = np.stack(
                [np.linalg.norm(selected_chart, axis=-1), np.arctan2(selected_chart[:, 1], selected_chart[:, 0])],
                axis=-1
            )

            # Visualize radial coordinates
            color_array = np.full((self.charts.shape[0], 4), fill_value=[1, 1, 1, 0.75])
            chart = selected_chart[:, 0]
            chart = (chart - chart.min()) / (chart.max() - chart.min())
            colors = cm.get_cmap("Reds")(chart)
            color_array[self.charts[chart_idx, :, 0] != np.inf] = colors
            trimesh.PointCloud(self.triangle_mesh.vertices, colors=color_array).show()

            # Visualize angular coordinates
            color_array = np.full((self.charts.shape[0], 4), fill_value=[1, 1, 1, 0.75])
            chart = selected_chart[:, 1]
            chart = (chart - chart.min()) / (chart.max() - chart.min())
            colors = cm.get_cmap("Greens")(chart)
            color_array[self.charts[chart_idx, :, 0] != np.inf] = colors
            trimesh.PointCloud(self.triangle_mesh.vertices, colors=color_array).show()

        # Create plot
        fig, ax = plt.subplots()

        # Read chart
        chart = self.charts[chart_idx]
        chart = chart[chart[:, 0] != np.inf]

        # Scatter plot
        polygons = PolyCollection(self.chart_triangles[chart_idx], alpha=0.4, edgecolors="red")
        ax.add_collection(polygons)

        # Circle
        if show_statistics:
            # Min radius
            circle_min = Circle(
                xy=(0., 0.),
                radius=float(self.min_chart_radius),
                color="blue",
                alpha=0.75,
                fill=False,
                linestyle='-',
                label="Min. radius"
            )
            ax.add_patch(circle_min)

            # Max radius
            circle_max = Circle(
                xy=(0., 0.),
                radius=float(self.max_chart_radius),
                color="red",
                alpha=0.75,
                fill=False,
                linestyle='-',
                label="Max. radius"
            )
            ax.add_patch(circle_max)

            # Average radius
            circle_avg = Circle(
                xy=(0., 0.),
                radius=float(self.avg_chart_radius),
                color="green",
                alpha=0.75,
                fill=False,
                linestyle='-.',
                label="Avg. radius"
            )
            ax.add_patch(circle_avg)

            # Median radius
            circle_median = Circle(
                xy=(0., 0.),
                radius=float(self.median_chart_radius),
                color="orange",
                alpha=0.75,
                fill=False,
                linestyle='-.',
                label="Median radius"
            )
            ax.add_patch(circle_median)

            eps = 0.05 * self.max_radius
            ax.set_xlim([-self.max_radius - eps, self.max_radius + eps])
            ax.set_ylim([-self.max_radius - eps, self.max_radius + eps])

            fig.subplots_adjust(right=0.79)
            fig.legend(loc="lower right", bbox_to_anchor=(1.0, 0.5), fontsize="small")
        else:
            ax.set_xlim([chart[:, 0].min(), chart[:, 0].max()])
            ax.set_ylim([chart[:, 1].min(), chart[:, 1].max()])

        # Misc
        ax.set_title(f"meta data: origin idx {chart_idx} - max-radius {self.max_radius} - method {self.method}")
        plt.grid()
        plt.show()

    def determine_barycentric_coordinates(self, n_radial, n_angular, radius):
        self.barycentric_coordinates[(n_radial, n_angular)] = compute_barycentric_coordinates(
            self,
            n_radial=n_radial,
            n_angular=n_angular,
            radius=radius
        )
