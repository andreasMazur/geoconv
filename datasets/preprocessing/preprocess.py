from datasets.preprocessing.barycentric_coords import create_kernel_matrix, barycentric_coordinates
from datasets.preprocessing.discrete_gpc import discrete_gpc

import scipy
import os
import trimesh
import pyshot
import numpy as np
import shutil
import sys
import tqdm


def preprocess(directory,
               target_dir,
               gpc_max_radius,
               kernel_size,
               use_c,
               eps,
               shuffle=True,
               reference_path=None):
    # Note that this directory will be deleted (recursively!) when preprocessing is finished.
    # Only the similar named zip-file will be kept.
    shuffled_meshes_dir = f"{target_dir}/shuffled_meshes"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]
    amt_files = len(file_list)

    if shuffle:
        print("Shuffling original meshes...")
        for file in tqdm.tqdm(file_list):
            if not os.path.exists(shuffled_meshes_dir):
                os.makedirs(shuffled_meshes_dir)

            object_mesh = trimesh.load_mesh(f"{directory}/{file}")
            object_mesh_vertices = np.copy(object_mesh.vertices)
            object_mesh_faces = np.copy(object_mesh.faces)

            shuffled_row_indices = np.arange(object_mesh.vertices.shape[0])
            np.random.shuffle(shuffled_row_indices)

            object_mesh_vertices = object_mesh_vertices[shuffled_row_indices]
            for face in object_mesh_faces:
                face[0] = np.where(shuffled_row_indices == face[0])[0]
                face[1] = np.where(shuffled_row_indices == face[1])[0]
                face[2] = np.where(shuffled_row_indices == face[2])[0]
            shuffled_object_mesh = trimesh.Trimesh(vertices=object_mesh_vertices, faces=object_mesh_faces)
            shuffled_object_mesh.export(f"{shuffled_meshes_dir}/{file}")

    if reference_path:
        print("Compute ground-truth for shape correspondence")
        query_dir = shuffled_meshes_dir if shuffle else directory
        reference_mesh = trimesh.load(reference_path)
        ref_mesh_kd_tree = scipy.spatial.KDTree(reference_mesh.vertices)
        for file in tqdm.tqdm(file_list):
            object_mesh = trimesh.load_mesh(f"{query_dir}/{file}")
            label_matrix = np.zeros(
                shape=(object_mesh.vertices.shape[0], reference_mesh.vertices.shape[0]), dtype=np.int32
            )
            for vertex_idx, vertex in enumerate(object_mesh.vertices):
                _, gt_idx = ref_mesh_kd_tree.query(vertex)
                label_matrix[vertex_idx, gt_idx] = 1.
            np.save(f"{target_dir}/GT_{file[:-4]}.npy", label_matrix)

    print("Preprocessing...")
    processing_dir = shuffled_meshes_dir if shuffle else directory
    for file_num, file in enumerate(file_list):
        sys.stdout.write(f"\rPreprocessing file {file}")
        object_mesh = trimesh.load_mesh(f"{processing_dir}/{file}")

        # Compute shot descriptor
        sys.stdout.write(f"\rComputing SHOT-descriptors...")
        descr = pyshot.get_descriptors(
            np.array(object_mesh.vertices),
            np.array(object_mesh.faces, dtype=np.int64),
            radius=100,
            local_rf_radius=.1,
            min_neighbors=3,
            n_bins=32
        )
        np.save(f"{target_dir}/SHOT_{file[:-4]}.npy", descr)

        # Compute local GPC-systems
        sys.stdout.write(f"\rComputing local GPC-systems..")
        local_gpc_systems = discrete_gpc(
            object_mesh, gpc_max_radius, eps, use_c, tqdm_msg=f"File {file_num + 1}/{amt_files}"
        )
        np.save(f"{target_dir}/GPC_{file[:-4]}.npy", local_gpc_systems)

        # Compute Barycentric coordinates
        sys.stdout.write(f"\rComputing Barycentric coordinates...")
        kernel = create_kernel_matrix(n_radial=kernel_size[0], n_angular=kernel_size[1], radius=gpc_max_radius-0.01)
        bary_coords = barycentric_coordinates(
            local_gpc_systems, kernel, object_mesh, tqdm_msg=f"File {file_num + 1}/{amt_files}"
        )
        np.save(f"{target_dir}/BC_{file[:-4]}.npy", bary_coords)

    shutil.make_archive(target_dir, "zip", target_dir)
    shutil.rmtree(target_dir)
    print("Preprocessing finished.")
