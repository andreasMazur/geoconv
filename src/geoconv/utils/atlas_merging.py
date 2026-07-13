from geoconv.preprocessing.atlas import Atlas

import numpy as np
import warnings


def merge_assertions(atlas_1, atlas_2):
    """Checks whether two atlases can be merged.

    Parameters
    ----------
    atlas_1: Atlas
        The first atlas for merging.
    atlas_2: Atlas
        The second atlas for merging.
    """
    # max_radius: float
    #     The maximal radius of charts during chart-computation.
    assert np.isclose(atlas_1.max_radius, atlas_2.max_radius), "The atlases vary in their maximum radius."

    # method: str
    #     The method to use to compute geodesic distances and angular direction. Select from ['dgpc', 'fmm', 'hdm].
    assert atlas_1.method == atlas_2.method, "The atlases vary in their selected charting method."

    # processes: int
    #     The concurrent processes for computed the charts.
    if atlas_1.processes != atlas_2.processes:
        warnings.warn(
            f"The atlases vary in their selected number of concurrent processes. Henceforth, the atlas uses "
            f"{min(atlas_1.processes, atlas_2.processes)} concurrent processes for computations."
        )

    # triangle_mesh: trimesh.Trimesh
    #     The shape for which charts are calculated.
    assert np.array_equal(atlas_1.triangle_mesh.vertices, atlas_2.triangle_mesh.vertices), (
        "The vertices vary between the underlying triangle meshes of the provided atlases."
    )
    assert np.array_equal(atlas_1.triangle_mesh.faces, atlas_2.triangle_mesh.faces), (
        "The faces vary between the underlying triangle meshes of the provided atlases."
    )

    # original_geodesic_diameter: float
    #     The original geodesic diameter of a chart.
    assert np.isclose(atlas_1.original_geodesic_diameter, atlas_2.original_geodesic_diameter), (
        "The original geodesic diameters of the underlying meshes are different."
    )

    # chart_indices: np.ndarray | None
    #     The indices of origin vertices around which charts are computed. If 'None', all origin vertices are used.
    assert atlas_1.chart_indices is not None, "Atlas 1 has no chart indices given."
    assert atlas_2.chart_indices is not None, "Atlas 2 has no chart indices given."
    compare_chart_indices = np.meshgrid(atlas_1.chart_indices, atlas_2.chart_indices)
    assert (compare_chart_indices[0] != compare_chart_indices[1]).all(), "Atlas 1 and 2 share equal chart indices."

    # charts: np.ndarray
    #     The computed charts.
    assert atlas_1.charts.shape[1] == atlas_2.charts.shape[1], (
        f"Charts in atlas 1 and 2 vary in their amount of considered neighbors - "
        f"atlas_1 shape: {atlas_1.charts.shape}, atlas_2 shape: {atlas_2.charts.shape}"
    )

    # barycentric_coordinates: dict
    #     A dictionary that contains barycentric coordinates that are computed with the given charts.
    assert set(atlas_1.barycentric_coordinates.keys()) == set(atlas_2.barycentric_coordinates.keys()), (
        "Atlas 1 and 2 contain different barycentric coordinates."
    )


def merge_atlases(atlas_1, atlas_2):
    """Merges the properties of two atlases. Assumes that both store information for the same shape.

    Parameters
    ----------
    atlas_1: Atlas
        The first atlas for merging.
    atlas_2: Atlas
        The second atlas for merging.

    Return
    ------
    Atlas:
        The merged atlas.
    """
    # Check whether atlases can be merged
    merge_assertions(atlas_1, atlas_2)

    # Meta information
    merged_max_radius = atlas_1.max_radius
    merged_method = atlas_1.method
    merged_processes = min(atlas_1.processes, atlas_2.processes)

    # Remember mesh
    merged_triangle_mesh = atlas_1.triangle_mesh.copy()
    merged_original_geodesic_diameter = atlas_1.original_geodesic_diameter

    ##########################
    # Merge all chart indices
    ##########################
    merged_chart_indices = np.concatenate([atlas_1.chart_indices, atlas_2.chart_indices], axis=0)
    sorting_indices = merged_chart_indices.argsort()
    merged_chart_indices = merged_chart_indices[sorting_indices]

    # Translate original chart indices into indices for chart- and BC-arrays
    index_map = {idx: i for i, idx in enumerate(merged_chart_indices)}
    atlas_1_chart_rows = np.array([index_map[idx] for idx in atlas_1.chart_indices])
    atlas_2_chart_rows = np.array([index_map[idx] for idx in atlas_2.chart_indices])

    ########################################################
    # Merge local charts (already in cartesian coordinates)
    ########################################################
    merged_charts = np.full(shape=(merged_chart_indices.shape[0], atlas_1.charts.shape[1], 2), fill_value=np.inf)
    merged_charts[atlas_1_chart_rows] = atlas_1.charts
    merged_charts[atlas_2_chart_rows] = atlas_2.charts

    #################################
    # Compute chart radii statistics
    #################################
    merged_chart_radii = np.array(
        [
            (distances[distances != np.inf].max() if distances.min() < np.inf else np.inf)
            for distances in np.linalg.norm(merged_charts, axis=-1)
        ]
    )
    mask = merged_chart_radii != np.inf
    merged_max_chart_radius = merged_chart_radii[mask].max()
    merged_min_chart_radius = merged_chart_radii[mask].min()
    merged_avg_chart_radius = merged_chart_radii[mask].mean()
    merged_std_chart_radius = merged_chart_radii[mask].std()
    merged_median_chart_radius = np.median(merged_chart_radii[mask])

    ################################################################
    # Merge x-axis indices / information about reference directions
    ################################################################
    merged_x_axes_indices = np.concatenate([atlas_1.x_axes_indices, atlas_2.x_axes_indices], axis=0)
    merged_x_axes_indices = merged_x_axes_indices[sorting_indices]

    ############################
    # Merge faces and triangles
    ############################
    merged_chart_faces = {**atlas_1.chart_faces, **atlas_2.chart_faces}
    merged_chart_triangles = {**atlas_1.chart_triangles, **atlas_2.chart_triangles}

    ################################
    # Merge barycentric coordinates
    ################################
    merged_barycentric_coordinates = {}
    for k in atlas_1.barycentric_coordinates.keys():
        merged_barycentric_coordinates[k] = np.concatenate(
            [atlas_1.barycentric_coordinates[k], atlas_2.barycentric_coordinates[k]], axis=0
        )[sorting_indices]

    # Merge custom arrays
    merged_custom_arrays = {"chart_indices": merged_chart_indices}

    ######################
    # Create atlas object
    ######################
    atlas = Atlas.__new__(Atlas)

    # Set meta information
    atlas.max_radius = merged_max_radius
    atlas.method = merged_method
    atlas.processes = merged_processes
    atlas.chart_indices = merged_chart_indices
    atlas.original_geodesic_diameter = merged_original_geodesic_diameter

    # Set chart statistics
    atlas.max_chart_radius = merged_max_chart_radius
    atlas.min_chart_radius = merged_min_chart_radius
    atlas.avg_chart_radius = merged_avg_chart_radius
    atlas.std_chart_radius = merged_std_chart_radius
    atlas.median_chart_radius = merged_median_chart_radius

    # Set triangle mesh and charts
    atlas.triangle_mesh = merged_triangle_mesh
    atlas.charts = merged_charts
    atlas.x_axes_indices = merged_x_axes_indices
    atlas.chart_radii = merged_chart_radii
    atlas.chart_faces = merged_chart_faces
    atlas.chart_triangles = merged_chart_triangles

    # Set barycentric coordinates
    atlas.barycentric_coordinates = merged_barycentric_coordinates

    # Set parallel transport angles
    atlas.parallel_transport = np.array([-1.])

    # Set custom arrays
    atlas.custom_arrays = merged_custom_arrays

    # Return instantiated atlas
    return atlas
