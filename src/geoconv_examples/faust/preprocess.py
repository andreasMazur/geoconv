from geoconv.preprocessing.barycentric_coordinates import compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.common import compute_gpc_systems
from geoconv.utils.data_generator import zip_file_generator
from geoconv.utils.misc import find_largest_one_hop_dist

import shutil
import pyshot
import numpy as np
import os
import json
import trimesh


def preprocess(faust_path, output_path, processes, zip_when_done=True):
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
        compute_gpc_systems(shape, output_dir, processes=processes)

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

    # Compute BC
    compute_bc(output_path)

    if zip_when_done:
        print("Zipping..")
        shutil.make_archive(base_name=output_path, format="zip", root_dir=output_path)
        shutil.rmtree(output_path)
        print("Done.")


def compute_bc(preprocess_dir, inverse_order=False):
    # Get average template radius
    gpc_system_radii = []
    preprocess_dir_temp = f"{preprocess_dir}/MPI-FAUST"
    for instance in os.listdir(f"{preprocess_dir_temp}/training/registrations"):
        shape_path = f"{preprocess_dir_temp}/training/registrations/{instance}"
        with open(f"{shape_path}/preprocess_properties.json") as properties_file:
            properties = json.load(properties_file)
            gpc_system_radii.append(properties["gpc_system_radius"])
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

    shapes = os.listdir(f"{preprocess_dir_temp}/training/registrations")
    shapes.sort()
    step = -1 if inverse_order else 1
    for instance in shapes[::step]:
        shape_path = f"{preprocess_dir_temp}/training/registrations/{instance}"

        # Load GPC-systems for current mesh
        gpc_systems = GPCSystemGroup(object_mesh=trimesh.load_mesh(f"{shape_path}/normalized_mesh.stl"))
        gpc_systems.load(f"{shape_path}/gpc_systems")

        # Compute barycentric coordinates
        for (n_radial, n_angular, template_radius) in template_configurations:
            bc = compute_barycentric_coordinates(
                gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=template_radius
            )
            np.save(f"{shape_path}/BC_{n_radial}_{n_angular}_{template_radius}.npy", bc)
    print(f"Barycentric coordinates done.")
