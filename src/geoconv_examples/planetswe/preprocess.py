from geoconv.preprocessing.atlas import Atlas, load_atlas

from geoconv_examples.planetswe.dataset import create_planetswe_sphere

import numpy as np
import time
import os
import shutil


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
    # Check whether zip-file already exists
    if os.path.isfile(f"{save_path}.zip"):
        print(f"File already exists: {save_path}.zip. Skipping...")
        return
    else:
        print(f"Computing dataset: {save_path}.zip")

    # Make sure saving folder exists
    os.makedirs(save_path, exist_ok=True)

    # Create a spherical triangular mesh
    sphere_mesh = create_planetswe_sphere(path, normalization_method=normalization_method, processes=processes)

    # Misc
    max_chart_radius_str = f"{max_chart_radius}".replace(".", "_")

    # Compute the atlas
    chart_indices_chunked = np.split(np.arange(sphere_mesh.vertices.shape[0]), chunks)
    chart_radii = []
    x_axis_indices = []
    for chunk_idx, chart_indices in enumerate(chart_indices_chunked):
        print(f"Currently computing the atlas for chunk {chunk_idx}: vertices {chart_indices[0]} - {chart_indices[-1]}")

        # Check whether file already exists
        atlas_save_path = f"{save_path}/{method}_{max_chart_radius_str}_{chart_indices[0]}_{chart_indices[-1]}.hdf5"
        if os.path.exists(atlas_save_path):
            continue

        atlas = Atlas(
            triangle_mesh=sphere_mesh,
            max_radius=max_chart_radius,
            method=method,
            normalization_method=None,
            processes=processes,
            chart_indices=chart_indices
        )
        chart_radii.extend(atlas.chart_radii.tolist())
        x_axis_indices.extend(atlas.x_axes_indices.tolist())
        save_atlas(atlas, atlas_save_path=atlas_save_path)

    # Compute barycentric coordinates
    for chart_indices in chart_indices_chunked:
        for (n_radial, n_angular) in template_resolutions:
            # Load atlas
            atlas_save_path = f"{save_path}/{method}_{max_chart_radius_str}_{chart_indices[0]}_{chart_indices[-1]}"
            if os.path.isdir(atlas_save_path):
                continue
            atlas = load_atlas(atlas_save_path)

            # Compute barycentric coordinates
            atlas.determine_barycentric_coordinates(
                n_radial=n_radial,
                n_angular=n_angular,
                radius=np.median(chart_radii),
                processes=processes
            )

            # Compute parallel transport angles
            atlas.determine_parallel_transport_angles(source_vertex_indices=atlas.chart_indices)

            # Save training data
            atlas.save_training_data(atlas_save_path)

            # Cleanup old atlas file
            os.remove(f"{atlas_save_path}.hdf5")

    # Zip everything
    print("Zipping...")
    shutil.make_archive(
        base_name=save_path,
        format="zip",
        root_dir=save_path
    )

    # Remove old logging dir
    shutil.rmtree(save_path)
