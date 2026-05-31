from geoconv.preprocessing.atlas import Atlas
from geoconv_examples.planetswe.dataset import create_planetswe_sphere

import numpy as np


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

    # Compute the atlas
    for chart_indices in np.split(np.arange(sphere_mesh.vertices.shape[0]), chunks):
        atlas = Atlas(
            triangle_mesh=sphere_mesh,
            max_radius=max_chart_radius,
            method=method,
            normalization_method=normalization_method,
            processes=processes,
            chart_indices=chart_indices
        )

        # Compute barycentric coordinates
        for (n_radial, n_angular) in template_resolutions:
            atlas.determine_barycentric_coordinates(
                n_radial=n_radial,
                n_angular=n_angular,
                radius=np.median(atlas.chart_radii.tolist())
            )
            # Save atlas
            atlas.save_training_data(f"{save_path}_{method}_{chart_indices[0]}_{chart_indices[-1]}")
