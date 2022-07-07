from preprocessing.barycentric_coords import create_kernel_matrix, barycentric_coordinates
from preprocessing.discrete_gpc import discrete_gpc

import os
import trimesh
import pyshot
import numpy as np
import shutil
import sys
import open3d as o3d
import tqdm


def preprocess(directory,
               target_dir,
               gpc_max_radius,
               kernel_size,
               use_c,
               eps,
               poisson_depth,
               poisson_lin_fit):
    # Note that this directory will be deleted (recursively!) when preprocessing is finished.
    # Only the similar named zip-file will be kept.
    sub_meshes_dir = f"{target_dir}/sub_meshes"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]
    amt_files = len(file_list)

    print("Sub-meshing original meshes...")
    for file in tqdm.tqdm(file_list):
        if not os.path.exists(sub_meshes_dir):
            os.makedirs(sub_meshes_dir)

        # Sub-mesh the given registrations
        object_mesh = o3d.io.read_triangle_mesh(f"{directory}/{file}")
        object_mesh.compute_vertex_normals()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = object_mesh.vertices
        point_cloud.normals = object_mesh.vertex_normals

        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd=point_cloud, depth=poisson_depth, linear_fit=poisson_lin_fit
        )[0]
        # o3d.visualization.draw_geometries([poisson_mesh[0]])
        o3d.io.write_triangle_mesh(f"{sub_meshes_dir}/sm_{file}", poisson_mesh)

    print("Preprocessing sub-meshes...")
    for file_num, file in enumerate(file_list):
        sys.stdout.write(f"\rPreprocessing file {file}")
        object_mesh = trimesh.load_mesh(f"{sub_meshes_dir}/sm_{file}")
        # point_cloud = trimesh.points.PointCloud(object_mesh.vertices)
        # point_cloud.show()

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


if __name__ == "__main__":
    FAUST_REGISTRATIONS = "/home/andreas/Uni/Masterarbeit/MPI-FAUST/training/registrations"
    TARGET_DIR = "../dataset/MPI_FAUST/preprocessed_registrations"
    GPC_MAX_RADIUS = 0.05
    KERNEL_SIZE = (2, 4)
    USE_C = True
    EPS = 0.000001
    POISSON_DEPTH = 7
    POISSON_LIN_FIT = True
    preprocess(
        FAUST_REGISTRATIONS, TARGET_DIR, GPC_MAX_RADIUS, KERNEL_SIZE, USE_C, EPS, POISSON_DEPTH, POISSON_LIN_FIT
    )
