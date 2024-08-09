from geoconv.preprocessing.wrapper import sample_surface
from geoconv.utils.data_generator import zip_file_generator

import shutil


def preprocess(modelnet_path,
               output_path,
               count,
               manifold_plus_executable,
               target_amount_faces=3000,
               zip_when_done=True,
               class_names=None):
    # Initialize shape generator
    shape_generator = zip_file_generator(
        modelnet_path,
        file_type="off",
        manifold_plus_executable=manifold_plus_executable,
        target_amount_faces=target_amount_faces,
        return_filename=True,
        shape_path_contains=class_names,
        normalize=False  # normalize during GPC-system computation to store original geodesic diameter
    )

    # Sample from each shape and save them
    for shape, shape_path in shape_generator:
        output_dir = f"{output_path}/{shape_path}"[:-4]
        sample_surface(shape, count, output_dir)

    # Zip preprocessed directory
    if zip_when_done:
        print("Zipping..")
        shutil.make_archive(base_name=output_path, format="zip", root_dir=output_path)
        shutil.rmtree(output_path)
        print("Done.")
