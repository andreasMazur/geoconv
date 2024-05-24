from geoconv.preprocessing.barycentric_coordinates import compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.misc import normalize_mesh, find_largest_one_hop_dist
from geoconv_examples.shapenet.data.shapenet_generator import up_shapenet_generator

import numpy as np
import os
import json
import shutil


def manifold_plus(shapenet_path, manifold_plus_executable, target_dir, synset_ids=None, depth=8):
    """Applies manifold plus to the ShapeNet dataset"""
    for synset_id in synset_ids:
        # Check if zip-file already exists for synset
        zip_file = f"{target_dir}/{synset_id}"
        if not os.path.isfile(f"{zip_file}.zip"):
            shapenet_generator = up_shapenet_generator(shapenet_path, return_filename=True, synset_ids=[synset_id])
            # 'shape_path' given w.r.t. ShapeNet-directory as root
            for shape, shape_path in shapenet_generator:
                # 'output_shape_path': where to store the repaired mesh (synset_id repetition for subsequent zipping)
                output_shape_path = f"{target_dir}/{synset_id}/{shape_path}"
                if not os.path.isfile(output_shape_path):
                    # Create shape directory
                    dir_name = os.path.dirname(output_shape_path)
                    os.makedirs(dir_name, exist_ok=True)

                    # Create temporary file for manifold+ algorithm
                    in_file = f"{dir_name}/model_normalized_temp.obj"
                    shape.export(in_file)

                    # Create output file
                    out_file = f"{dir_name}/model_normalized.obj"

                    # Manifold plus algorithm
                    if np.asarray(shape.as_open3d.get_non_manifold_edges()).shape[0] > 0:
                        os.system(f"{manifold_plus_executable} --input {in_file} --output {out_file} --depth {depth}")
                    else:
                        shape.export(out_file)

                    # Remove temporary file
                    os.remove(in_file)
            # Zip synset-id directory to save memory
            print(f"Manifold+ done. Zipping synset '{synset_id}'..")
            shutil.make_archive(base_name=zip_file, format="zip", root_dir=zip_file)
            shutil.rmtree(zip_file)
            print("Done.")
        else:
            print(f"Zip-file already exists: {zip_file}")


def compute_gpc_systems(target_dir, synset_ids, down_sample=6000, processes=1):
    for synset_id in synset_ids:
        shapenet_generator = up_shapenet_generator(
            target_dir,
            return_filename=True,
            remove_non_manifold_edges=True,
            down_sample=down_sample,
            synset_ids=[synset_id]
        )
        for shape, shape_path in shapenet_generator:
            # 'output_shape_path': where to store the preprocessed mesh (synset_id repetition for subsequent zipping)
            output_shape_path = f"{target_dir}/{synset_id}/{shape_path}"
            if not os.path.isfile(output_shape_path):
                # Only preprocess meshes with more than 100 vertices
                if shape.vertices.shape[0] >= 100:
                    # Create shape directory
                    dir_name = os.path.dirname(output_shape_path)
                    os.makedirs(dir_name, exist_ok=True)

                    # 0.) Store mesh
                    shape.export(f"{dir_name}/model_normalized.obj")

                    # 1.) Normalize shape
                    properties_file_path = f"{dir_name}/preprocess_properties.json"
                    gpc_system_radius = None
                    if os.path.isfile(properties_file_path):
                        with open(properties_file_path, "r") as properties_file:
                            properties = json.load(properties_file)
                            shape, geodesic_diameter = normalize_mesh(
                                shape, geodesic_diameter=properties["geodesic_diameter"]
                            )
                            gpc_system_radius = properties["gpc_system_radius"]
                    else:
                        shape, geodesic_diameter = normalize_mesh(shape)

                    # 2.) Compute GPC-systems
                    gpc_systems_path = f"{dir_name}/gpc_systems"
                    if not os.path.exists(gpc_systems_path):
                        gpc_systems = GPCSystemGroup(shape, processes=processes)
                        gpc_system_radius = find_largest_one_hop_dist(shape) if gpc_system_radius is None else gpc_system_radius
                        gpc_systems.compute(u_max=gpc_system_radius)
                        gpc_systems.save(gpc_systems_path)
                    else:
                        gpc_systems = GPCSystemGroup(shape, processes=processes)
                        gpc_systems.load(gpc_systems_path)

                    # 3.) Log preprocess properties
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
        print(f"Preprocessing '{synset_id}' done. Zipping..")
        zip_file = f"{target_dir}/{synset_id}"
        os.remove(f"{zip_file}.zip")
        shutil.make_archive(base_name=zip_file, format="zip", root_dir=zip_file)
        shutil.rmtree(zip_file)
        print("Done.")


def preprocess_shapenet(n_radial,
                        n_angular,
                        kernel_radius,
                        shapenet_path,
                        target_dir,
                        manifold_plus_executable,
                        synset_ids,
                        down_sample=6000,
                        depth=8,
                        processes=1):
    ####################################
    # Convert meshes to manifold meshes
    ####################################
    manifold_plus(
        shapenet_path, manifold_plus_executable, target_dir=target_dir, synset_ids=synset_ids, depth=depth
    )

    ######################
    # Compute GPC-systems
    ######################
    compute_gpc_systems(target_dir, synset_ids, down_sample=down_sample, processes=processes)

    ########################################
    # TODO: Compute barycentric coordinates
    ########################################
