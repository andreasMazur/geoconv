from geoconv.preprocessing.barycentric_coordinates import compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.misc import normalize_mesh, find_largest_one_hop_dist, get_faces_of_edge
from geoconv_examples.shapenet.data.shapenet_generator import up_shapenet_generator, up_unpacked_shapenet_generator

import numpy as np
import os
import json
import shutil
import trimesh
import zipfile


def remove_nme(mesh):
    """Removes non-manifold edges by removing all their faces."""
    # Check if non-manifold edges exist
    non_manifold_edges = np.asarray(mesh.as_open3d.get_non_manifold_edges())
    if non_manifold_edges.shape[0] > 0:
        # Compute mask that removes non-manifold edges and all their faces
        face_mask = np.full(mesh.faces.shape[0], True)
        for edge in non_manifold_edges:
            sorted_edge, edge_faces = get_faces_of_edge(edge, mesh)
            for edge_f in edge_faces:
                update_mask = np.logical_not((edge_f == mesh.faces).all(axis=-1))
                face_mask = np.logical_and(face_mask, update_mask)
        # Remove non-manifold edges and faces with mask
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[face_mask])
    return mesh


def down_sample_mesh(mesh, target_number_of_triangles):
    """Down-samples the mesh."""
    mesh = mesh.as_open3d.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)
    trimesh.repair.fill_holes(mesh)
    return mesh


def repair_shape(shape,
                 dir_name,
                 manifold_plus_executable,
                 depth=8,
                 down_sample=None,
                 remove_non_manifold_edges=True):
    """Repairs the given shape."""
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

    # Remove temporary file and non-".obj"-files
    os.remove(in_file)
    for file_name in [f"{dir_name}/{file_name}" for file_name in os.listdir(dir_name) if file_name[-3:] != "obj"]:
        os.remove(file_name)

    # Load repaired shape
    shape = trimesh.load_mesh(out_file)

    # Down-sample shape
    if down_sample is not None and shape.faces.shape[0] > down_sample:
        shape = down_sample_mesh(shape, down_sample)

    # Remove non-manifold edges
    if remove_non_manifold_edges:
        shape = remove_nme(shape)

    # Update out file
    os.remove(out_file)
    shape.export(out_file)

    # Remove non-".obj"-files
    for file_name in [f"{dir_name}/{file_name}" for file_name in os.listdir(dir_name) if file_name[-3:] != "obj"]:
        os.remove(file_name)

    return shape


def compute_gpc_systems(shapenet_root,
                        target_root,
                        manifold_plus_executable,
                        synset_ids,
                        down_sample=6000,
                        processes=1,
                        min_vertices=100,
                        depth=8):
    """Computes GPC-systems."""
    for synset_id in synset_ids:
        # Check if zip for synset already exists
        if not os.path.isfile(f"{target_root}/{synset_id}.zip"):
            shapenet_generator = up_shapenet_generator(shapenet_root, return_filename=True, synset_ids=[synset_id])
            for shape, shape_path in shapenet_generator:
                # output_shape_path: where to store the preprocessed mesh (synset_id repetition for subsequent zipping)
                dir_name = os.path.dirname(f"{target_root}/{synset_id}/{shape_path}")

                # If properties file exists, GPC-computation ran through and we can skip addition GPC-computation.
                properties_file_path = f"{dir_name}/preprocess_properties.json"
                if not os.path.isfile(properties_file_path):
                    # Create shape directory
                    os.makedirs(dir_name, exist_ok=True)

                    # Repair shape (manifold+-algorithm + down-sample + remove nme)
                    shape = repair_shape(
                        shape,
                        dir_name,
                        manifold_plus_executable,
                        depth=depth,
                        down_sample=down_sample,
                        remove_non_manifold_edges=True
                    )

                    # Only compute GPC-systems for meshes with more than 100 vertices
                    if shape.vertices.shape[0] >= min_vertices and shape.faces.shape[0] > 0:
                        # 1.) Normalize shape
                        shape, geodesic_diameter = normalize_mesh(shape)

                        # 2.) Compute GPC-systems
                        gpc_systems_path = f"{dir_name}/gpc_systems"
                        if not os.path.exists(gpc_systems_path):
                            gpc_systems = GPCSystemGroup(shape, processes=processes)
                            gpc_system_radius = find_largest_one_hop_dist(shape)
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
                    else:
                        print(f"Could not repair: {shape_path}")
                        shutil.rmtree(dir_name)
                else:
                    print(f"Found preprocess-properties file: {properties_file_path}")
            print(f"Preprocessing '{synset_id}' done. Zipping..")
            zip_file = f"{target_root}/{synset_id}"
            shutil.make_archive(base_name=zip_file, format="zip", root_dir=zip_file)
            shutil.rmtree(zip_file)
            print("Done.")


def compute_bc(dataset_root, synset_ids, n_radial, n_angular, kernel_radius, processes=1):
    """Computes barycentric coordinates."""
    for synset_id in synset_ids:
        # Unzip ShapeNet-synset with GPC-systems
        print(f"Unzipping synset '{synset_id}' for computing barycentric coordinates..")
        synset_zip = f"{dataset_root}/{synset_id}.zip"
        with zipfile.ZipFile(synset_zip, "r") as zip_ref:
            zip_ref.extractall(synset_zip[:-4])
        os.remove(synset_zip)
        print("Done.")

        shapenet_generator = up_unpacked_shapenet_generator(dataset_root, return_filename=True, synset_ids=[synset_id])
        for shape, shape_path in shapenet_generator:
            # 'output_bc_path': where to store the preprocessed mesh (synset_id repetition for subsequent zipping)
            shape_dir = os.path.dirname(shape_path)
            output_bc_path = f"{dataset_root}/{shape_dir}/BC_{n_radial}_{n_angular}_{kernel_radius}.npy"
            if not os.path.isfile(output_bc_path):
                with open(f"{dataset_root}/{shape_dir}/preprocess_properties.json", "r") as properties_file:
                    properties = json.load(properties_file)
                    geodesic_diameter = properties["geodesic_diameter"]

                gpc_systems = GPCSystemGroup(normalize_mesh(shape, geodesic_diameter)[0], processes=processes)
                gpc_systems.load(f"{dataset_root}/{shape_dir}/gpc_systems")

                # Compute barycentric coordinates
                bc = compute_barycentric_coordinates(
                    gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=kernel_radius
                )
                np.save(output_bc_path, bc)
        print(f"Computing barycentric coordinates for synset '{synset_id}' done. Zipping..")
        zip_file = f"{dataset_root}/{synset_id}"
        shutil.make_archive(base_name=zip_file, format="zip", root_dir=zip_file)
        shutil.rmtree(zip_file)
        print("Done.")


def get_gpc_stats(dataset_root):
    """Get GPC-system stats of ShapeNet."""
    shapenet_generator = up_shapenet_generator(dataset_root, return_properties=True)
    avg_radius = 0
    for idx, (shape, properties) in enumerate(shapenet_generator):
        avg_radius += properties["gpc_system_radius"]
    avg_radius = avg_radius / (idx + 1)
    return avg_radius


def preprocess_shapenet(n_radial,
                        n_angular,
                        shapenet_path,
                        target_dir,
                        manifold_plus_executable,
                        synset_ids,
                        kernel_radius=None,
                        down_sample=6000,
                        depth=8,
                        processes=1,
                        min_vertices=100):
    """Wrapper function to preprocess ShapeNet."""
    ######################
    # Compute GPC-systems
    ######################
    compute_gpc_systems(
        shapenet_path,
        target_dir,
        manifold_plus_executable,
        synset_ids,
        down_sample=down_sample,
        processes=processes,
        min_vertices=min_vertices,
        depth=depth
    )

    ##################################
    # Compute barycentric coordinates
    ##################################
    if kernel_radius is None:
        gpc_system_radius = get_gpc_stats(target_dir)
        kernel_radius = gpc_system_radius * 0.75
    compute_bc(target_dir, synset_ids, n_radial, n_angular, kernel_radius)

