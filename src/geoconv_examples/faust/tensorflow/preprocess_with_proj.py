import json

from geoconv.tensorflow.layers import BarycentricCoordinates
from geoconv.utils.data_generator import zip_file_generator
from geoconv.utils.misc import normalize_mesh, find_largest_one_hop_dist
from geoconv_examples.faust.geodesic_diameters import GEODESIC_DIAMETERS

import numpy as np
import shutil
import pyshot
import os


def proj_preprocess(faust_path, output_path, template_configurations):
    shape_generator = zip_file_generator(
        faust_path,
        file_type="ply",
        manifold_plus_executable=None,
        target_amount_faces=None,
        return_filename=True,
        shape_path_contains=["registrations"],
        normalize=False  # Set to False here and normalize with given geodesic diameter later
    )

    for s_id, (shape, shape_path) in enumerate(shape_generator):
        output_dir = f"{output_path}/{shape_path}"[:-4]
        os.makedirs(output_dir, exist_ok=True)

        shape, _ = normalize_mesh(shape, geodesic_diameter=GEODESIC_DIAMETERS[s_id])

        # Compute SHOT-descriptor
        radius = find_largest_one_hop_dist(shape) * 2.5
        shot_descriptor = pyshot.get_descriptors(
            shape.vertices,
            shape.faces,
            radius=radius,
            local_rf_radius=radius,
            min_neighbors=10,
            n_bins=16,
            double_volumes_sectors=True,
            use_interpolation=True,
            use_normalization=True
        )
        np.save(f"{output_dir}/SIGNAL.npy", shot_descriptor)

        # Compute barycentric coordinates
        for n_radial, n_angular, template_radius in template_configurations:
            print(
                f"Currently processing:"
                f" {shape_path} with n_radial={n_radial},"
                f" n_angular={n_angular},"
                f" template_radius={template_radius}"
            )
            bc_layer = BarycentricCoordinates(
                n_radial=n_radial,
                n_angular=n_angular,
                neighbors_for_lrf=20,
                projection_neighbors=20,
            )
            bc_layer.adapt(template_radius=template_radius)
            bc = bc_layer(shape.vertices.reshape(1, -1, 3))[0]
            np.save(f"{output_dir}/BC_{n_radial}_{n_angular}_{template_radius}.npy", bc)

        # Create preprocess properties file so that loader can find the data
        with open(f"{output_dir}/preprocess_properties.json", "w") as properties_file:
            json.dump({}, properties_file, indent=4)

    print("Zipping..")
    shutil.make_archive(base_name=output_path, format="zip", root_dir=output_path)
    shutil.rmtree(output_path)
    print("Done.")
