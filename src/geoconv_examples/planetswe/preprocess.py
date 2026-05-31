from geoconv.preprocessing.atlas import Atlas
from geoconv_examples.planetswe.dataset import create_planetswe_sphere

import numpy as np


def preprocess(path,
               save_path,
               max_chart_radius,
               method,
               normalization_method,
               template_resolutions,
               processes=1):
    """Computes barycentric coordinates for the sphere for planetswe."""
    # Create a spherical triangular mesh
    sphere_mesh = create_planetswe_sphere(path)

    # Compute the atlas
    atlas = Atlas(
        triangle_mesh=sphere_mesh,
        max_radius=max_chart_radius,
        method=method,
        normalization_method=normalization_method,
        processes=processes
    )

    # Compute barycentric coordinates
    for (n_radial, n_angular) in template_resolutions:
        atlas.determine_barycentric_coordinates(
            n_radial=n_radial,
            n_angular=n_angular,
            radius=np.median(atlas.chart_radii.tolist())
        )
        # Save atlas
        atlas.save_training_data(save_path)
