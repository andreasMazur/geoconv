from geoconv.preprocessing.barycentric_coordinates import compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.misc import normalize_mesh, find_largest_one_hop_dist
from geoconv_examples.shapenet.data.shapenet_generator import up_shapenet_generator, up_shapenet_generator_unpacked

import numpy as np
import os
import json
import shutil


def manifold_plus(shapenet_path, manifold_plus_executable, output_path, temp_dir="./temp", synset_ids=None, depth=8):
    """Applies manifold plus to the ShapeNet dataset"""
    shapenet_generator = up_shapenet_generator(shapenet_path, return_filename=True, synset_ids=synset_ids)
    for shape, shape_path in shapenet_generator:
        # Filter file directory name (-> synset/model_id)
        shape_path = "/" + "/".join(shape_path.split("/")[-4:-1])

        # Create temporary file for manifold+ algorithm
        temp_fn = temp_dir + shape_path
        if not os.path.exists(temp_fn):
            os.makedirs(temp_fn)
        in_file = temp_fn + "/model_normalized.obj"
        shape.export(in_file)

        # Create output file
        output_fn = output_path + shape_path
        if not os.path.exists(output_fn):
            os.makedirs(output_fn)
        out_file = output_fn + "/model_normalized.obj"

        # Manifold plus algorithm
        if np.asarray(shape.as_open3d.get_non_manifold_edges()).shape[0] > 0:
            os.system(f"{manifold_plus_executable} --input {in_file} --output {out_file} --depth {depth}")
        else:
            shape.export(out_file)
    shutil.rmtree(temp_dir)


def preprocess_shapenet(n_radial,
                        n_angular,
                        kernel_radius,
                        shapenet_path,
                        target_dir,
                        manifold_plus_executable,
                        synset_ids=None,
                        down_sample=6000,
                        depth=8,
                        processes=1,
                        compute_bc=True):
    ####################################
    # Convert meshes to manifold meshes
    ####################################
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        manifold_plus(
            shapenet_path, manifold_plus_executable, output_path=target_dir, synset_ids=synset_ids, depth=depth
        )

    ##################################
    # Compute barycentric coordinates
    ##################################
    shapenet_generator = up_shapenet_generator_unpacked(
        target_dir, return_filename=True, remove_non_manifold_edges=True, down_sample=down_sample, synset_ids=synset_ids
    )
    for shape, shape_path in shapenet_generator:
        # Only preprocess meshes with more than 100 vertices
        if shape.vertices.shape[0] >= 100:
            # Get shape directory
            shape_directory = "/".join(shape_path.split("/")[:-1])

            # 1) Normalize shape
            properties_file_path = f"{shape_directory}/preprocess_properties.json"
            gpc_system_radius = None
            if os.path.isfile(properties_file_path):
                with open(properties_file_path, "r") as properties_file:
                    properties = json.load(properties_file)
                    shape, geodesic_diameter = normalize_mesh(shape, geodesic_diameter=properties["geodesic_diameter"])
                    gpc_system_radius = properties["gpc_system_radius"]
            else:
                shape, geodesic_diameter = normalize_mesh(shape)

            # 2.) Compute GPC-systems
            gpc_systems_path = f"{shape_directory}/gpc_systems"
            if not os.path.exists(gpc_systems_path):
                gpc_systems = GPCSystemGroup(shape, processes=processes)
                gpc_system_radius = find_largest_one_hop_dist(shape) if gpc_system_radius is None else gpc_system_radius
                gpc_systems.compute(u_max=gpc_system_radius)
                gpc_systems.save(gpc_systems_path)
            else:
                gpc_systems = GPCSystemGroup(shape, processes=processes)
                gpc_systems.load(gpc_systems_path)

            # 3.) Compute barycentric coordinates
            if compute_bc:
                bary_coords = compute_barycentric_coordinates(
                    gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=kernel_radius
                )
                np.save(f"{shape_directory}/barycentric_coordinates.npy", bary_coords)

            # 4.) Log preprocess properties
            with open(properties_file_path, "w") as properties_file:
                json.dump(
                    {
                        "non_manifold_edges": np.asarray(shape.as_open3d.get_non_manifold_edges()).shape[0],
                        "gpc_system_radius": gpc_system_radius,
                        "geodesic_diameter": geodesic_diameter,
                        "kernel_radius": kernel_radius if compute_bc else None
                    },
                    properties_file,
                    indent=4
                )
