from geoconv.utils.common import compute_gpc_systems
from geoconv.utils.data_generator import zip_file_generator

import shutil


def preprocess(modelnet_path, output_path, manifold_plus_executable, down_sample, processes):
    # Initialize shape generator
    shape_generator = zip_file_generator(
        modelnet_path,
        file_type="off",
        manifold_plus_executable=manifold_plus_executable,
        down_sample=down_sample,
        return_filename=True
    )

    # Compute GPC-systems
    for shape, shape_path in shape_generator:
        print(f"*** Preprocessing: '{shape_path}'")
        # Remove file-ending from folder name
        compute_gpc_systems(shape, f"{output_path}/{shape_path}"[:-4], processes=processes)

    # Zip results
    print(f"Preprocessing done. Zipping..")
    shutil.make_archive(base_name=output_path, format="zip", root_dir=output_path)
    shutil.rmtree(output_path)
    print("Done.")
