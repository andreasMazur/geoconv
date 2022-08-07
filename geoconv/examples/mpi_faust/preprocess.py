from geoconv.preprocessing.barycentric_coords import create_kernel_matrix, barycentric_coordinates
from geoconv.preprocessing.discrete_gpc import discrete_gpc

import open3d as o3d
import os
import tqdm
import numpy as np
import scipy
import shutil
import trimesh
import pyshot


def preprocess(directory, target_dir, sub_sample_amount, reference_mesh):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]

    ############################
    # Sub-sample reference mesh
    ############################
    reference_mesh = o3d.io.read_triangle_mesh(reference_mesh)
    reference_mesh.compute_vertex_normals()
    reference_mesh = reference_mesh.sample_points_poisson_disk(sub_sample_amount)
    ref_mesh_kd_tree = scipy.spatial.KDTree(reference_mesh.points)

    with tqdm.tqdm(total=len(file_list)) as pbar:
        for file_no, file in enumerate(file_list):
            pbar.set_postfix({"Step": "Sub-sample the original meshes"})
            ############
            # Load mesh
            ############
            mesh = trimesh.load_mesh(f"{directory}/{file}")

            pbar.set_postfix({"Step": "Ground-truth computation"})
            ################################################
            # Compute ground-truth for shape correspondence
            ################################################
            label_matrix = np.zeros(
                shape=(np.array(mesh.vertices).shape[0], np.array(reference_mesh.points).shape[0]), dtype=np.int8
            )
            for vertex_idx, vertex in enumerate(mesh.vertices):
                _, gt_idx = ref_mesh_kd_tree.query(vertex)
                label_matrix[vertex_idx, gt_idx] = 1.
            label_matrix = scipy.sparse.csc_array(label_matrix)
            np.save(f"{target_dir}/GT_{file[:-4]}.npy", label_matrix)

            pbar.set_postfix({"Step": "Compute SHOT descriptors"})
            ###########################
            # Compute SHOT descriptors
            ###########################
            descriptors = pyshot.get_descriptors(
                np.array(mesh.vertices),
                np.array(mesh.faces, dtype=np.int64),
                radius=100,
                local_rf_radius=.1,
                min_neighbors=3,
                n_bins=8,
                double_volumes_sectors=False,
                use_interpolation=True,
                use_normalization=True,
            ).astype(np.float32)
            np.save(f"{target_dir}/SHOT_{file[:-4]}.npy", descriptors)

            pbar.set_postfix({"Step": "Compute local GPC-systems"})
            ############################
            # Compute local GPC-systems
            ############################
            local_gpc_systems = discrete_gpc(
                mesh, u_max=0.05, eps=.000001, use_c=True, tqdm_msg=f"File {file_no} - Compute local GPC-systems"
            ).astype(np.float64)

            pbar.set_postfix({"Step": "Compute Barycentric coordinates"})
            ##################################
            # Compute Barycentric coordinates
            ##################################
            kernel = create_kernel_matrix(n_radial=2, n_angular=4, radius=0.04)
            bary_coords = barycentric_coordinates(
                local_gpc_systems, kernel, mesh, tqdm_msg=f"File {file_no} - Compute Barycentric coordinates"
            )
            np.save(f"{target_dir}/BC_{file[:-4]}.npy", bary_coords)

            pbar.update(1)

    shutil.make_archive(target_dir, "zip", target_dir)
    shutil.rmtree(target_dir)
    print("Preprocessing finished.")
