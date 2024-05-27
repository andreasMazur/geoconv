from geoconv.preprocessing.barycentric_coordinates import compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.common import compute_gpc_systems
from geoconv.utils.data_generator import zip_file_generator

from tqdm import tqdm

import trimesh
import shutil
import os
import json
import numpy as np


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
            compute_gpc_systems(
                shape, f"{output_path}/{synset}/{'/'.join(shape_path.split('/')[:-1])}", processes=processes
            )

        # Compute BC
        compute_bc(f"{output_path}/{synset}", n_radial=5, n_angular=8)


def compute_bc(preprocess_dir, n_radial, n_angular):
    # Get average template radius
    template_radii = []
    preprocess_dir_temp = f"{preprocess_dir}/{preprocess_dir.split('/')[-1]}"
    for shape_id in tqdm(os.listdir(preprocess_dir_temp), postfix="Computing template radius for BC.."):
        shape_path = f"{preprocess_dir_temp}/{shape_id}/models"
        with open(f"{shape_path}/preprocess_properties.json") as properties_file:
            properties = json.load(properties_file)
            template_radii.append(properties["gpc_system_radius"] * 0.75)
    template_radius = np.array(template_radii).mean()

    # Compute BC
    for shape_id in os.listdir(preprocess_dir_temp):
        shape_path = f"{preprocess_dir_temp}/{shape_id}/models"

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
