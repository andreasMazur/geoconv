from geoconv.preprocessing.wrapper import compute_gpc_systems_wrapper, compute_bc_wrapper
from geoconv.utils.data_generator import zip_file_generator
from geoconv.utils.misc import find_largest_one_hop_dist

import shutil
import pyshot
import numpy as np


def preprocess(faust_path, output_path, processes, zip_when_done=True, compute_gpc=True, compute_bc=True):
    assert compute_gpc or compute_bc, "You must either set 'compute_gpc' or 'compute_bc' to 'True'."

    if compute_gpc:
        # Initialize shape generator
        shape_generator = zip_file_generator(
            faust_path,
            file_type="ply",
            manifold_plus_executable=None,
            down_sample=None,
            return_filename=True,
            shape_path_contains=["registrations"]
        )

        # Compute GPC-systems
        for shape, shape_path in shape_generator:
            print(f"*** Preprocessing: '{shape_path}'")
            # Remove file-ending from folder name
            output_dir = f"{output_path}/{shape_path}"[:-4]

            # Compute GPC-systems
            compute_gpc_systems_wrapper(shape, output_dir, processes=processes)

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

    if compute_bc:
        # Compute BC
        compute_bc_wrapper(
            preprocess_dir=output_path,
            template_sizes=[(3, 6), (5, 8)],
            scales=[0.75, 1.0, 1.25],
            load_compressed_gpc_systems=True,
            processes=processes
        )

    if zip_when_done:
        print("Zipping..")
        shutil.make_archive(base_name=output_path, format="zip", root_dir=output_path)
        shutil.rmtree(output_path)
        print("Done.")
