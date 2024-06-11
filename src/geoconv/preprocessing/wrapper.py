from geoconv.preprocessing.barycentric_coordinates import compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.misc import normalize_mesh, find_largest_one_hop_dist

from tqdm import tqdm

import shutil
import json
import os
import numpy as np
import trimesh


def compute_gpc_systems_wrapper(shape, output_dir, processes=1):
    """Wrapper function that computes all GPC systems for one given shape.

    Parameters
    ----------
    shape: trimesh.Trimesh
        A manifold mesh.
    output_dir: str
        The directory where the GPC-systems shall be stored.
    processes: int
        The amount of processes to be used concurrently.

    Returns
    -------
    bool:
        Whether preprocessing has been successful.
    """
    # 0.) Check whether file already exist. If so, skip computing GPC-systems.
    if not os.path.isfile(f"{output_dir}/preprocess_properties.json"):
        # 1.) Create output dir if not existent
        os.makedirs(output_dir, exist_ok=True)

        # 2.) Normalize shape
        try:
            shape, geodesic_diameter = normalize_mesh(shape)
        except RuntimeError:
            print(f"{output_dir} crashed during normalization. Skipping preprocessing.")
            shutil.rmtree(output_dir)
            return

        # 3.) Compute GPC-systems
        gpc_systems = GPCSystemGroup(shape, processes=processes)
        gpc_system_radius = find_largest_one_hop_dist(shape)
        gpc_systems.compute(u_max=gpc_system_radius)
        gpc_systems.save(f"{output_dir}/gpc_systems")

        # 4.) Log preprocess properties
        properties_file_path = f"{output_dir}/preprocess_properties.json"
        with open(properties_file_path, "w") as properties_file:
            json.dump(
                {
                    "non_manifold_edges": np.asarray(shape.as_open3d.get_non_manifold_edges()).shape[0],
                    "gpc_system_radius": gpc_system_radius,
                    "original_geodesic_diameter": geodesic_diameter,
                    "amount_gpc_systems": gpc_systems.object_mesh_gpc_systems.shape[0]
                },
                properties_file,
                indent=4
            )

        # 5.) Export preprocessed mesh
        shape.export(f"{output_dir}/normalized_mesh.stl")
        return True
    else:
        print(f"{output_dir}/preprocess_properties.json already exists. Skipping preprocessing.")
        return False


def compute_bc_wrapper(preprocess_dir, template_sizes, scales=None, load_compressed_gpc_systems=True):
    """

    Parameters
    ----------
    preprocess_dir
    template_sizes
    load_compressed_gpc_systems

    Returns
    -------

    """
    # Get average template radius as well as most seen GPC-systems in a shape
    shape_directories, gpc_system_radii, most_gpc_systems = [], [], 0
    for (dir_path, sub_dir_names, dir_files) in os.walk(preprocess_dir):
        # Search for shape-directories
        if "preprocess_properties.json" in dir_files:
            shape_directories.append(dir_path)
            # Get GPC-system radius and amount of GPC-systems from current shape
            with open(f"{dir_path}/preprocess_properties.json") as properties_file:
                properties = json.load(properties_file)
                gpc_system_radii.append(properties["gpc_system_radius"])
                n_gpc_systems = properties["amount_gpc_systems"]
                most_gpc_systems = n_gpc_systems if n_gpc_systems > most_gpc_systems else most_gpc_systems
    # Compute final average GPC-system radius
    avg_gpc_system_radius = np.array(gpc_system_radii).mean()

    # Configure template size
    if scales is None:
        scales = [0.75, 1., 1.25]
    template_configurations = [
        template_size + (avg_gpc_system_radius * scale,) for template_size in template_sizes for scale in scales
    ]

    # Compute barycentric coordinates
    shape_directories.sort(key=lambda directory_name: directory_name.split("/")[-1])
    for shape_dir in shape_directories:
        # Load GPC-systems for current mesh
        gpc_systems = GPCSystemGroup(object_mesh=trimesh.load_mesh(f"{shape_dir}/normalized_mesh.stl"))
        gpc_systems.load(f"{shape_dir}/gpc_systems", load_compressed=load_compressed_gpc_systems)

        # Compute barycentric coordinates
        for (n_radial, n_angular, template_radius) in template_configurations:
            bc_file_name = f"{shape_dir}/BC_{n_radial}_{n_angular}_{template_radius}.npy"
            if not os.path.isfile(bc_file_name):
                bc = compute_barycentric_coordinates(
                    gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=template_radius
                )
                np.save(bc_file_name, bc)

    return template_configurations, most_gpc_systems
