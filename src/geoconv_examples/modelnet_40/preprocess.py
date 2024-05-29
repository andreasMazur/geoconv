from geoconv.preprocessing.barycentric_coordinates import compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.common import compute_gpc_systems
from geoconv.utils.data_generator import zip_file_generator

from tqdm import tqdm

import trimesh
import shutil
import json
import numpy as np
import os


def preprocess(modelnet_path, output_path, manifold_plus_executable, down_sample, processes, class_names=None):
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
    for shape, shape_path in shape_generator:
        print(f"*** Preprocessing: '{shape_path}'")
        # Remove file-ending from folder name
        output_dir = f"{output_path}/{shape_path}"[:-4]
        if class_names is None:
            compute_gpc_systems(shape, output_dir, processes=processes)
        elif shape_path.split("/")[1] in class_names:
            compute_gpc_systems(shape, output_dir, processes=processes)

    # Compute barycentric coordinates
    compute_bc(output_path, n_radial=5, n_angular=8)


def compute_bc(preprocess_dir, n_radial, n_angular):
    # Get average template radius
    template_radii = []
    preprocess_dir_temp = f"{preprocess_dir}/ModelNet40"
    for shape_class in tqdm(os.listdir(preprocess_dir_temp), postfix="Computing template radius for BC.."):
        for split in ["test", "train"]:
            for instance in os.listdir(f"{preprocess_dir_temp}/{shape_class}/{split}/"):
                shape_path = f"{preprocess_dir_temp}/{shape_class}/{split}/{instance}"

                with open(f"{shape_path}/preprocess_properties.json") as properties_file:
                    properties = json.load(properties_file)
                    template_radii.append(properties["gpc_system_radius"] * 0.75)
    template_radius = np.array(template_radii).mean()

    # Compute BC
    for shape_class in os.listdir(preprocess_dir_temp):
        for split in ["test"]:  # "train",
            for instance in os.listdir(f"{preprocess_dir_temp}/{shape_class}/{split}/"):
                shape_path = f"{preprocess_dir_temp}/{shape_class}/{split}/{instance}"

                gpc_systems = GPCSystemGroup(object_mesh=trimesh.load_mesh(f"{shape_path}/normalized_mesh.stl"))
                gpc_systems.load(f"{shape_path}/gpc_systems")

                bc = compute_barycentric_coordinates(
                    gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=template_radius
                )
                np.save(f"{shape_path}/BC_{n_radial}_{n_angular}_{template_radius}.npy", bc)

    print(f"Barycentric coordinates done. Zipping..")
    shutil.make_archive(base_name=preprocess_dir, format="zip", root_dir=preprocess_dir)
    shutil.rmtree(preprocess_dir)
    print("Done.")
