from geoconv.preprocessing.wrapper import sample_surface
from geoconv.utils.data_generator import zip_file_generator

import shutil
import sys


def preprocess(modelnet_path,
               output_path,
               count,
               zip_when_done=True,
               class_names=None):
    # Initialize shape generator,
    shape_generator = zip_file_generator(
        zipfile_path=modelnet_path,
        file_type="off",
        manifold_plus_executable=None,
        target_amount_faces=None,
        return_filename=True,
        min_vertices=0,
        shape_path_contains=class_names,
        remove_non_manifold_edges=False,
        normalize=False,
        repair_shapes=True
    )

    # Sample from each shape and save them
    for shape, shape_path in shape_generator:
        sys.stdout.write(f"\rProcessing shape: {shape_path}")
        output_dir = f"{output_path}/{shape_path}"[:-4]
        sample_surface(shape, count, output_dir)

    # Zip preprocessed directory
    if zip_when_done:
        print("Zipping..")
        shutil.make_archive(base_name=output_path, format="zip", root_dir=output_path)
        shutil.rmtree(output_path)
        print("Done.")
