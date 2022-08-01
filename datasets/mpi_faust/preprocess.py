from datasets.preprocessing.barycentric_coords import create_kernel_matrix, barycentric_coordinates
from datasets.preprocessing.discrete_gpc import discrete_gpc

import open3d as o3d
import os
import tqdm
import numpy as np
import pyshot
import scipy
import shutil
import trimesh


def preprocess(directory, target_dir, sub_sample_amount, reference_mesh, sub_samples_per_mesh=3):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]

    reference_mesh = o3d.io.read_triangle_mesh(reference_mesh)
    ref_mesh_kd_tree = scipy.spatial.KDTree(reference_mesh.vertices)

    with tqdm.tqdm(total=len(file_list) * sub_samples_per_mesh) as pbar:
        for file in file_list:
            for sample in range(sub_samples_per_mesh):

                pbar.set_postfix({"Step": "Sub-sample the original meshes"})
                #################################
                # Sub-sample the original meshes
                #################################
                mesh = o3d.io.read_triangle_mesh(f"{directory}/{file}")
                mesh.compute_vertex_normals()
                point_cloud = mesh.sample_points_poisson_disk(sub_sample_amount)
                distances = point_cloud.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radii = [avg_dist * 1.3, avg_dist * 1.4, avg_dist * 1.5]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    point_cloud, o3d.utility.DoubleVector(radii)
                )

                pbar.set_postfix({"Step": "Ground-truth computation"})
                ################################################
                # Compute ground-truth for shape correspondence
                ################################################
                label_matrix = np.zeros(
                    shape=(np.array(mesh.vertices).shape[0], np.array(reference_mesh.vertices).shape[0]), dtype=np.int32
                )
                for vertex_idx, vertex in enumerate(mesh.vertices):
                    _, gt_idx = ref_mesh_kd_tree.query(vertex)
                    label_matrix[vertex_idx, gt_idx] = 1.
                np.save(f"{target_dir}/GT_{file[:-4]}_{sample}.npy", label_matrix)

                pbar.set_postfix({"Step": "Compute SHOT descriptors"})
                ###########################
                # Compute SHOT descriptors
                ###########################
                # descriptors = pyshot.get_descriptors(
                #     np.array(mesh.vertices),
                #     np.array(mesh.triangles, dtype=np.int64),
                #     radius=100,
                #     local_rf_radius=.1,
                #     min_neighbors=3,
                #     n_bins=32
                # )
                np.save(f"{target_dir}/SHOT_{file[:-4]}_{sample}.npy", np.array(mesh.vertices))

                pbar.set_postfix({"Step": "Compute local GPC-systems"})
                ############################
                # Compute local GPC-systems
                ############################
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)
                local_gpc_systems = discrete_gpc(mesh, u_max=0.05, eps=.000001, use_c=True)
                np.save(f"{target_dir}/GPC_{file[:-4]}_{sample}.npy", local_gpc_systems)

                pbar.set_postfix({"Step": "Compute Barycentric coordinates"})
                ##################################
                # Compute Barycentric coordinates
                ##################################
                kernel = create_kernel_matrix(n_radial=2, n_angular=4, radius=0.04)
                bary_coords = barycentric_coordinates(local_gpc_systems, kernel, mesh)

                missing_amt_gpc_systems = sub_sample_amount - bary_coords.shape[0]
                missing_bary_coords = np.zeros((missing_amt_gpc_systems, 4, 2, 6))
                bary_coords = np.concatenate([bary_coords, missing_bary_coords])
                np.save(f"{target_dir}/BC_{file[:-4]}_{sample}.npy", bary_coords)

                pbar.update(1)

    shutil.make_archive(target_dir, "zip", target_dir)
    shutil.rmtree(target_dir)
    print("Preprocessing finished.")
