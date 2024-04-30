from geoconv.preprocessing.barycentric_coordinates import compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.misc import normalize_mesh, find_largest_one_hop_dist, get_faces_of_edge
from geoconv_examples.graspable.data.data_set import raw_data_generator

import shutil
import numpy as np
import trimesh
import os
import json


def preprocess_data(data_path, target_dir, temp_dir=None, processes=1, n_radial=5, n_angular=8):
    if temp_dir is None:
        temp_dir = "./temp_meshes"

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Create raw data generator
    rd_generator = raw_data_generator(data_path)

    # Normalize meshes
    for mesh_idx, (mesh, _) in enumerate(rd_generator):
        # Define file names for normed vertices and faces
        normalized_v_name = f"{temp_dir}/vertices_{mesh_idx}.npy"
        normalized_f_name = f"{temp_dir}/faces_{mesh_idx}.npy"

        # Center and normalize mesh to unit geodesic diameter
        bad_edges = np.array(mesh.as_open3d.get_non_manifold_edges())
        face_mask = np.full(mesh.faces.shape[0], True)
        for edge in bad_edges:
            sorted_edge, edge_faces = get_faces_of_edge(edge, mesh)
            for edge_f in edge_faces:
                update_mask = np.logical_not((edge_f == mesh.faces).all(axis=-1))
                face_mask = np.logical_and(face_mask, update_mask)
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[face_mask])
        normed_mesh, geodesic_diameter = normalize_mesh(mesh)

        # Log geodesic diameter
        with open(f"{target_dir}/geodesic_diameters.txt", "a") as diameters_file:
            diameters_file.write(f"{geodesic_diameter}\n")

        # Save normalized mesh
        np.save(normalized_v_name, np.asarray(normed_mesh.vertices))
        np.save(normalized_f_name, np.asarray(normed_mesh.faces))

    # Find GPC-system radius
    amount_meshes, gpc_radius = mesh_idx, .0
    for mesh_idx in range(amount_meshes):
        # Load normalized mesh
        vertices = np.load(f"{temp_dir}/vertices_{mesh_idx}.npy")
        faces = np.load(f"{temp_dir}/faces_{mesh_idx}.npy")
        normed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Find largest one-hop neighborhood distance and check whether it is new GPC-system radius
        new_candidate = find_largest_one_hop_dist(normed_mesh)
        gpc_radius = new_candidate if new_candidate > gpc_radius else gpc_radius
    assert gpc_radius > 0., "No valid gpc radius found."

    # Set template radius
    kernel_radius = gpc_radius * 0.75
    print(f"GPC-system radius: {gpc_radius:.3f} | Kernel radius: {kernel_radius:.3f}")

    # Log GPC-system radius and kernel radius
    with open(f"{target_dir}/gpc_kernel_properties.json", "w") as properties_file:
        json.dump({"gpc_system_radius": gpc_radius, "kernel_radius": kernel_radius}, properties_file, indent=4)

    # Create raw data generator
    rd_generator = raw_data_generator(data_path, return_file_name=True)

    # Compute GPC-systems and barycentric coordinates
    for mesh_idx, (_, vertex_labels, file_name) in enumerate(rd_generator):
        # Define file names
        bc_name = f"{target_dir}/BC_{file_name}.npy"
        gt_name = f"{target_dir}/GT_{file_name}.npy"
        signal_name = f"{target_dir}/SIGNAL_{file_name}.npy"

        # Load normalized mesh
        vertices = np.load(f"{temp_dir}/vertices_{mesh_idx}.npy")
        faces = np.load(f"{temp_dir}/faces_{mesh_idx}.npy")
        normed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Save ground truth
        np.save(gt_name, vertex_labels)

        # Store mesh signal (3D coordinates)
        np.save(signal_name, vertices)

        # Compute GPC-systems
        gpc_systems = GPCSystemGroup(normed_mesh, processes=processes)
        gpc_systems.compute(u_max=gpc_radius)

        # Compute and save barycentric coordinates
        bary_coords = compute_barycentric_coordinates(
            gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=kernel_radius
        )
        np.save(bc_name, bary_coords)

    shutil.rmtree(temp_dir)
    shutil.make_archive(target_dir, "zip", target_dir)
    shutil.rmtree(target_dir)
    print("Preprocessing finished.")
