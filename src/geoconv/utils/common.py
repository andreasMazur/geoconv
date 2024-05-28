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

        # 4.) Export preprocessed mesh
        shape.export(f"{output_dir}/normalized_mesh.stl")
    else:
        print(f"{output_dir}/preprocess_properties.json already exists. Skipping preprocessing.")
