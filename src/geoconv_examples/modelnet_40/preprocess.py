from geoconv.preprocessing.wrapper import compute_gpc_systems_wrapper, compute_bc_wrapper
from geoconv.utils.data_generator import zip_file_generator

import shutil
import json


def preprocess(modelnet_path,
               output_path,
               manifold_plus_executable,
               down_sample,
               processes,
               class_names=None,
               zip_when_done=True):
    # Initialize shape generator
    shape_generator = zip_file_generator(
        modelnet_path,
        file_type="off",
        manifold_plus_executable=manifold_plus_executable,
        down_sample=down_sample,
        return_filename=True,
        shape_path_contains=class_names
    )

    # Compute GPC-systems
    preprocessed_shapes = 0
    for shape, shape_path in shape_generator:
        print(f"*** Preprocessing: '{shape_path}'")
        # Remove file-ending from folder name
        output_dir = f"{output_path}/{shape_path}"[:-4]
        if class_names is None:
            was_successful = compute_gpc_systems_wrapper(shape, output_dir, processes=processes)
        elif shape_path.split("/")[1] in class_names:
            was_successful = compute_gpc_systems_wrapper(shape, output_dir, processes=processes)
        preprocessed_shapes = preprocessed_shapes + 1 if was_successful else preprocessed_shapes

    # Compute barycentric coordinates
    template_configurations, most_gpc_systems = compute_bc_wrapper(
        preprocess_dir=output_path,
        template_sizes=[(3, 6), (5, 8)],
        scales=[0.75, 1.0, 1.25],
        load_compressed_gpc_systems=True,
        processes=processes
    )

    # Add preprocess information to dataset
    with open(f"{output_path}/dataset_properties.json", "a") as properties_file:
        temp_conf_dict = {"considered_classes": class_names, "most_gpc_systems": most_gpc_systems}
        for idx, tconf in enumerate(template_configurations):
            temp_conf_dict[f"{idx}"] = {
                "n_radial": tconf[0],
                "n_angular": tconf[1],
                "template_radius": tconf[2]
            }
        json.dump(temp_conf_dict, properties_file, indent=4)

    if zip_when_done:
        print("Zipping..")
        shutil.make_archive(base_name=output_path, format="zip", root_dir=output_path)
        shutil.rmtree(output_path)
        print("Done.")
