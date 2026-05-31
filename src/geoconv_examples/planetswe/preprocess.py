from geoconv.preprocessing.atlas import Atlas, load_atlas
from geoconv_examples.planetswe.dataset import create_planetswe_sphere

import numpy as np
import time


def save_atlas(atlas, atlas_save_path):
    is_saved = False
    tries = 0
    while not is_saved:
        try:
            atlas.save(atlas_save_path)
            is_saved = True
        except BlockingIOError:
            print(f"Trying to save atlas.. {tries}")
            tries += 1
            time.sleep(1)


def preprocess(path,
               save_path,
               max_chart_radius,
               method,
               normalization_method,
               template_resolutions,
               processes=1,
               chunks=16):
    """Computes barycentric coordinates for the sphere for planetswe."""
    # Create a spherical triangular mesh
    sphere_mesh = create_planetswe_sphere(path)
    max_chart_radius_str = f"{max_chart_radius}".replace(".", "_")

    # Compute the atlas
    chart_indices_chunked = np.split(np.arange(sphere_mesh.vertices.shape[0]), chunks)[:2]
    chart_radii = []
    for chunk_idx, chart_indices in enumerate(chart_indices_chunked):
        print(f"Currently computing the atlas for chunk {chunk_idx}: vertices {chart_indices[0]} - {chart_indices[-1]}")
        atlas = Atlas(
            triangle_mesh=sphere_mesh,
            max_radius=max_chart_radius,
            method=method,
            normalization_method=normalization_method,
            processes=processes,
            chart_indices=chart_indices
        )
        chart_radii.extend(atlas.chart_radii.tolist())
        save_atlas(
            atlas,
            atlas_save_path=f"{save_path}_{method}_{max_chart_radius_str}_{chart_indices[0]}_{chart_indices[-1]}"
        )

    # Compute barycentric coordinates
    for chart_indices in chart_indices_chunked:
        for (n_radial, n_angular) in template_resolutions:
            atlas = load_atlas(f"{save_path}_{method}_{max_chart_radius_str}_{chart_indices[0]}_{chart_indices[-1]}")
            atlas.determine_barycentric_coordinates(
                n_radial=n_radial,
                n_angular=n_angular,
                radius=np.median(chart_radii),
                processes=processes
            )
            atlas.save_training_data(f"{save_path}_{method}_{max_chart_radius_str}_{chart_indices[0]}_{chart_indices[-1]}")
