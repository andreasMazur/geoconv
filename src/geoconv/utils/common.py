from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.misc import normalize_mesh, find_largest_one_hop_dist

import shutil
import json
import os
import numpy as np


def compute_gpc_systems(shape, output_dir, processes=1):
    """Wrapper function that computes all GPC systems for one given shape.

    Parameters
    ----------
    shape: trimesh.Trimesh
        A manifold mesh.
    output_dir: str
        The directory where the GPC-systems shall be stored.
    processes: int
        The amount of processes to be used concurrently.
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
                    "geodesic_diameter": geodesic_diameter
                },
                properties_file,
                indent=4
            )

        # 5.) Export preprocessed mesh
        shape.export(f"{output_dir}/normalized_mesh.stl")
    else:
        print(f"{output_dir}/preprocess_properties.json already exists. Skipping preprocessing.")


def read_template_configurations(zipfile_path):
    """Reads the template configurations stored within a preprocessed dataset.

    Parameters
    ----------
    zipfile_path: str
        The path to the preprocessed dataset.

    Returns
    -------
    list:
        A list of tuples of the form (n_radial, n_angular, template_radius). These configurations have been
        found in the given zipfile.
    """
    # Load barycentric coordinates
    zip_file = np.load(zipfile_path)

    # Filter for barycentric coordinates files
    filtered_content = [file_name for file_name in zip_file.files if file_name[:2] == "BC"]
    filtered_content.sort()

    # Collect all found template configurations
    template_configurations = set()
    for bc_path in filtered_content:
        bc_properties = tuple(bc_path.split("_")[1:])
        template_configurations.add((int(bc_properties[0]), int(bc_properties[1]), float(bc_properties[2])))

    return list(template_configurations)
