from geoconv.preprocessing.barycentric_coordinates import compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.misc import normalize_mesh, find_largest_one_hop_dist

from multiprocessing import Pool

import shutil
import json
import os
import numpy as np
import trimesh
import math


def compute_gpc_systems_wrapper(shape, output_dir, processes=1, scale=1.):
    """Wrapper function that computes all GPC systems for one given shape.

    Parameters
    ----------
    shape: trimesh.Trimesh
        A manifold mesh.
    output_dir: str
        The directory where the GPC-systems shall be stored.
    processes: int
        The amount of processes to be used concurrently.
    scale: float
        A coefficient to scale the maximal distance of a GPC-system.

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
            return False

        # 3.) Compute GPC-systems
        gpc_systems = GPCSystemGroup(shape, processes=processes)
        gpc_system_radius = find_largest_one_hop_dist(shape) * scale
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


def compute_bc_wrapper(preprocess_dir,
                       template_sizes,
                       scales=None,
                       load_compressed_gpc_systems=True,
                       processes=10,
                       shape_path_contains=None):
    """Given a directory structure containing GPC-systems, this function computes corresponding barycentric coordinates.

    Parameters
    ----------
    preprocess_dir: str
        The path to the directory structure.
    template_sizes: list
        A list of tuples containing the wished template sizes for which barycentric coordinates shall be computed.
    scales: list
        A list of floats used for scaling the radii of the templates.
    load_compressed_gpc_systems: bool
        If 'True', assumes that GPC-systems are stored in the compressed format (cf. GPCSystemGroup.save()).
        Otherwise, assume that each GPC-system for a shape has its own directory.
    processes: int
        The amount of parallel processes to use.
    shape_path_contains: list
        A list of strings that is contained in the shape-path. If none of the contained strings are within the
        shape-path, then the shape is skipped.

    Returns
    -------
    bool:
        Whether all barycentric coordinates have been computed.
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

    # Split the list of all directories into multiple chunks
    shape_directories.sort(key=lambda directory_name: directory_name.split("/")[-1])
    if shape_path_contains is not None:
        shape_directories = [
            d for d in shape_directories if np.any([substring in d for substring in shape_path_contains])
        ]
    preprocessed_shapes = len(shape_directories)
    per_chunk = math.ceil(len(shape_directories) / processes)
    shape_directories = [shape_directories[i * per_chunk:(i * per_chunk) + per_chunk] for i in range(processes)]

    # Compute barycentric coordinates
    with Pool(processes=processes) as p:
        all_bc_computed = p.starmap(
            bc_helper, [(d, template_configurations, load_compressed_gpc_systems) for d in shape_directories]
        )

    # Check whether all barycentric coordinates have been computed
    if not np.all(all_bc_computed):
        # If not, return 'False' to indicate something went wrong.
        return False
    else:
        # If preprocess succeeded, add preprocess information to dataset
        with open(f"{preprocess_dir}/dataset_properties.json", "a") as properties_file:
            temp_conf_dict = {
                "preprocessed_shapes": preprocessed_shapes,
                "most_gpc_systems": most_gpc_systems,
                "template_configurations": {}
            }
            for idx, tconf in enumerate(template_configurations):
                temp_conf_dict["template_configurations"][f"{idx}"] = {
                    "n_radial": tconf[0], "n_angular": tconf[1], "template_radius": tconf[2]
                }
            json.dump(temp_conf_dict, properties_file, indent=4)
        # Return 'True' to indicate everything worked out.
        return True


def bc_helper(assigned_directories, template_configurations, load_compressed_gpc_systems):
    """Given a set of shape directories, compute barycentric coordinates for given template configurations.

    Parameters
    ----------
    assigned_directories: list
        A list containing paths to shape directories
    template_configurations: list
        A list containing template configurations (radial, angular) for which barycentric coordinates shall be computed
    load_compressed_gpc_systems: bool
        If 'True', assumes that GPC-systems are stored in the compressed format (cf. GPCSystemGroup.save()).
        Otherwise, assume that each GPC-system for a shape has its own directory.

    Returns
    -------
    bool:
        Whether all GPC-systems were successfully loaded (True = succeeded, False = failed) for BC-computation.
    """
    loading_succeeded = True
    for shape_dir in assigned_directories:
        gpc_systems = None
        for (n_radial, n_angular, template_radius) in template_configurations:
            bc_file_name = f"{shape_dir}/BC_{n_radial}_{n_angular}_{template_radius}.npy"
            # Only compute new BC-coordinates if nonexistent so far
            if not os.path.isfile(bc_file_name):

                # Load GPC-systems for current mesh
                if gpc_systems is None:
                    gpc_systems = GPCSystemGroup(object_mesh=trimesh.load_mesh(f"{shape_dir}/normalized_mesh.stl"))
                    try:
                        gpc_systems.load(f"{shape_dir}/gpc_systems", load_compressed=load_compressed_gpc_systems)
                    except RecursionError:
                        print(f"*** Recursion occurred error while loading GPC-systems of: {shape_dir}")
                        loading_succeeded = False
                        break

                # Compute barycentric coordinates
                bc = compute_barycentric_coordinates(
                    gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=template_radius
                )

                # Save barycentric coordinates
                np.save(bc_file_name, bc)
    return loading_succeeded
