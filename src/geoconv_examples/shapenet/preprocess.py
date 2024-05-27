from geoconv.utils.common import compute_gpc_systems
from geoconv.utils.data_generator import zip_file_generator

import shutil


def preprocess(shapenet_path, output_path, manifold_plus_executable, down_sample, processes, synsets):
    for synset in synsets:
        synset_dir = f"{shapenet_path}/{synset}"
        # Initialize shape generator
        shape_generator = zip_file_generator(
            f"{synset_dir}.zip",
            file_type="obj",
            manifold_plus_executable=manifold_plus_executable,
            down_sample=down_sample,
            return_filename=True
        )

        # Compute GPC-systems
        for shape, shape_path in shape_generator:
            print(f"*** Preprocessing: '{shape_path}'")
            # Remove file-ending from folder name
            compute_gpc_systems(shape, f"{output_path}/{synset}/{shape_path}"[:-4], processes=processes)

        # Zip results
        print(f"Preprocessing done. Zipping..")
        synset_output_path = f"{output_path}/{synset}"
        shutil.make_archive(base_name=synset_output_path, format="zip", root_dir=synset_output_path)
        shutil.rmtree(synset_output_path)
        print("Done.")
