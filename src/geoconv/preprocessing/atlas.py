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


def longest_axis_normalization(triangle_mesh):
    """Normalizes mesh by scaling its longest axis to one and moving its point of mass to zero.

    Parameters
    ----------
    triangle_mesh: trimesh.Trimesh
        The mesh to normalize.

    Returns
    -------
    trimesh.Trimesh:
        The normalized triangle mesh.
    """
    x_length = triangle_mesh.vertices[:, 0].max() - triangle_mesh.vertices[:, 0].min()
    y_length = triangle_mesh.vertices[:, 1].max() - triangle_mesh.vertices[:, 1].min()
    z_length = triangle_mesh.vertices[:, 2].max() - triangle_mesh.vertices[:, 2].min()
    normalized_vertices = triangle_mesh.vertices / np.max([x_length, y_length, z_length])
    normalized_vertices = normalized_vertices - np.mean(normalized_vertices, axis=0)
    return trimesh.Trimesh(vertices=normalized_vertices, faces=triangle_mesh.faces)


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
    """Loads an atlas.

    Parameters
    ----------
    filepath: str
        The filepath to the stored atlas.

    Returns
    -------
    Atlas:
        The loaded atlas.
    """
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

        # Load barycentric coordinates
        barycentric_coordinates = {}
        for template_res in f["barycentric_coordinates"].keys():
            template_res_key = tuple([int(x) for x in template_res.split("_")])
            barycentric_coordinates[template_res_key] = np.array(f["barycentric_coordinates"][template_res])

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
    atlas.barycentric_coordinates = barycentric_coordinates

    # Return instantiated atlas
    return atlas


class Atlas:
    """A class that computes, administrates and visualizes sets of local charts for one shape.

    Attributes
    ----------
    max_radius: float
        The maximal radius of charts during chart-computation.
    method: str
        The method to use to compute geodesic distances and angular direction. Select from ['dgpc', 'fmm', 'hdm].
    processes: int
        The concurrent processes for computed the charts.
    triangle_mesh: trimesh.Trimesh
        The shape for which charts are calculated.
    original_geodesic_diameter: float
        The original geodesic diameter of a chart.
    charts: np.ndarray
        The computed charts.
    chart_radii: np.ndarray
        An array of maximal geodesic distances for the computed charts.
    max_chart_radius: float
        The maximal observed geodesic distance among all charts.
    min_chart_radius: float
        The minimal observed geodesic distance among all charts.
    avg_chart_radius: float
        The average observed geodesic distance among all charts.
    std_chart_radius: float
        The standard deviation of observed geodesic distances among all charts.
    median_chart_radius: float
        The median observed geodesic distance among all charts.
    chart_faces: dict
        A dictionary that contains the faces that can be entirely described by local coordinates of charts.
    chart_triangles: dict
        A dictionary that contains the triangles that can be entirely described by local coordinates of charts.
    barycentric_coordinates: dict
        A dictionary that contains barycentric coordinates that are computed with the given charts.
    """
    def __init__(self, triangle_mesh, max_radius, method="hdm", normalization_method="hdm", processes=1):
        # Meta information
        self.max_radius = max_radius
        self.method = method
        self.processes = processes

        # Normalize mesh
        if normalization_method == "longest_axis":
            self.triangle_mesh = longest_axis_normalization(triangle_mesh)
            geodesic_diameter = -1.
        elif normalization_method is None:
            print("No shape normalization conducted since 'normalization_method = None'.")
            self.triangle_mesh = triangle_mesh
            geodesic_diameter = -1.
        else:
            self.triangle_mesh, geodesic_diameter = normalize_shape(
                triangle_mesh,
                method=normalization_method,
                processes=processes
            )
        self.original_geodesic_diameter = geodesic_diameter

        # Local charts
        self.charts = calculate_local_charts(
            self.triangle_mesh,
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
        """Saves the entire atlas.

        Parameters
        ----------
        filepath: str
            The location at which to store the atlas.
        """
        filepath = f"{filepath}.hdf5" if not filepath.endswith(".hdf5") else filepath
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

            # Save computed barycentric coordinates
            h5_bc_information = f.create_group("barycentric_coordinates")
            for (n_radial, n_angular), bc in self.barycentric_coordinates.items():
                h5_bc_information.create_dataset(f"{n_radial}_{n_angular}", data=bc, compression="gzip")

    def visualize_chart(self, chart_idx, visualize_3d=False, show_statistics=True, show_vertex_indices=False):
        """Visualizes one chart of the atlas.

        Parameters
        ----------
        chart_idx: int
            The index of the chart to visualize.
        visualize_3d: bool
            Whether to show the chart on the shape in 3D.
        show_statistics: bool
            Whether to include statistics in the plot.
        show_vertex_indices: bool
            Whether to include vertex indices at their corresponding positions in the plot.
        """
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

        # Annotate each point with its number
        if show_vertex_indices:
            cf = self.chart_faces[chart_idx]
            ct = self.chart_triangles[chart_idx]
            for f, t in zip(cf, ct):
                for idx in range(3):
                    plt.text(t[idx][0], t[idx][1], str(f[idx]), fontsize=8, ha="right", va="bottom")

        # Mark center
        ax.scatter(0., 0., color="black", s=10, label="Chart origin")

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

            eps = 0.01 * self.max_chart_radius
            ax.set_xlim([-self.max_chart_radius - eps, self.max_chart_radius + eps])
            ax.set_ylim([-self.max_chart_radius - eps, self.max_chart_radius + eps])

            fig.subplots_adjust(right=0.79)
            fig.legend(loc="lower right", bbox_to_anchor=(1.0, 0.5), fontsize="small")
        else:
            eps = 0.01 * chart[:, 0].max()
            ax.set_xlim([chart[:, 0].min() - eps, chart[:, 0].max() + eps])
            ax.set_ylim([chart[:, 1].min() - eps, chart[:, 1].max() + eps])

        # Misc
        ax.set_title(f"Config: origin idx {chart_idx} - max-radius {self.max_radius} - method {self.method}")
        plt.grid()
        plt.show()

    def determine_barycentric_coordinates(self, n_radial, n_angular, radius, processes=None):
        self.barycentric_coordinates[(n_radial, n_angular)] = compute_barycentric_coordinates(
            self,
            n_radial=n_radial,
            n_angular=n_angular,
            radius=radius,
            processes=self.processes if processes is None else processes
        )
