from geoconv.preprocessing.barycentric_coordinates import compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.preprocessing.wrapper import compute_gpc_systems
from geoconv.utils.data_generator import zip_file_generator

from tqdm import tqdm

import trimesh
import shutil
import json
import numpy as np
import os


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
            was_successful = compute_gpc_systems(shape, output_dir, processes=processes)
        elif shape_path.split("/")[1] in class_names:
            was_successful = compute_gpc_systems(shape, output_dir, processes=processes)
        preprocessed_shapes = preprocessed_shapes + 1 if was_successful else preprocessed_shapes

    # Compute barycentric coordinates
    template_configurations, most_gpc_systems = compute_bc(output_path)

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


def compute_bc(preprocess_dir, inverse_order=False, shape_classes=None):
    # Get average template radius and most gpc-systems among shapes
    gpc_system_radii, most_gpc_systems = [], 0
    preprocess_dir_temp = f"{preprocess_dir}/ModelNet40"
    for shape_class in tqdm(os.listdir(preprocess_dir_temp), postfix="Computing template radius for BC.."):
        for split in ["test", "train"]:
            shape_folders = os.listdir(f"{preprocess_dir_temp}/{shape_class}/{split}/")
            shape_folders.sort()
            for shape_folder in shape_folders:
                shape_path = f"{preprocess_dir_temp}/{shape_class}/{split}/{shape_folder}"
                with open(f"{shape_path}/preprocess_properties.json") as properties_file:
                    properties = json.load(properties_file)
                    gpc_system_radii.append(properties["gpc_system_radius"])
                    n_gpc_systems = properties["amount_gpc_systems"]
                    most_gpc_systems = n_gpc_systems if n_gpc_systems > most_gpc_systems else most_gpc_systems
    avg_gpc_system_radius = np.array(gpc_system_radii).mean()

    # Define template configurations
    template_configurations = [
        (3, 6, avg_gpc_system_radius * .75),
        (3, 6, avg_gpc_system_radius),
        (3, 6, avg_gpc_system_radius * 1.25),
        (5, 8, avg_gpc_system_radius * .75),
        (5, 8, avg_gpc_system_radius),
        (5, 8, avg_gpc_system_radius * 1.25)
    ]

    # Compute BC
    if shape_classes is None:
        shape_classes = os.listdir(preprocess_dir_temp)
    shape_classes.sort()
    step = -1 if inverse_order else 1
    for shape_class in shape_classes[::step]:
        for split in ["test", "train"]:
            shape_folders = os.listdir(f"{preprocess_dir_temp}/{shape_class}/{split}/")
            shape_folders.sort()
            for shape_folder in shape_folders:
                shape_path = f"{preprocess_dir_temp}/{shape_class}/{split}/{shape_folder}"

                # Load GPC-systems for current mesh
                gpc_systems = GPCSystemGroup(object_mesh=trimesh.load_mesh(f"{shape_path}/normalized_mesh.stl"))
                gpc_systems.load(f"{shape_path}/gpc_systems")

                # Compute barycentric coordinates
                for (n_radial, n_angular, template_radius) in template_configurations:
                    bc_file_name = f"{shape_path}/BC_{n_radial}_{n_angular}_{template_radius}.npy"
                    if not os.path.isfile(bc_file_name):
                        bc = compute_barycentric_coordinates(
                            gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=template_radius
                        )
                        np.save(bc_file_name, bc)
    print(f"Barycentric coordinates done.")
    return template_configurations, most_gpc_systems
