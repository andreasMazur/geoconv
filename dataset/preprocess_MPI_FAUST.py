from tqdm import tqdm

from barycentric_coordinates.barycentric_coords import create_kernel_matrix, barycentric_coordinates
from geodesic_polar_coordinates.discrete_gpc import discrete_gpc

import os
import trimesh
import pyshot
import numpy as np
import shutil


def compute_barycentric_coordinates(directory, target_dir, k_radius, k_radial_coords, k_angular_coords):
    """Computes Barycentric coordinates for the MPI-FAUST registration files."""

    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]
    amt_files = len(file_list)
    for file_num, file in enumerate(file_list[:1]):
        # Load object mesh and corresponding GPC-systems
        object_mesh = trimesh.load_mesh(f"{directory}/{file}")
        local_gpc_systems = np.load(f"{target_dir}/GPC_{file[:-4]}.npy")

        kernel = create_kernel_matrix(n_radial=k_radial_coords, n_angular=k_angular_coords, radius=k_radius)
        bary_coords = barycentric_coordinates(
            local_gpc_systems, kernel, object_mesh, tqdm_msg=f"File {file_num+1}/{amt_files}"
        )

        np.save(f"{target_dir}/BC_{file[:-4]}.npy", bary_coords)


def compute_gpc_systems(directory, target_dir, u_max, eps, use_c):
    """Computes local GPC-systems for the MPI-FAUST registration files."""

    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]
    amt_files = len(file_list)
    for file_num, file in enumerate(file_list[:1]):
        object_mesh = trimesh.load_mesh(f"{directory}/{file}")
        local_gpc_systems = discrete_gpc(object_mesh, u_max, eps, use_c, tqdm_msg=f"File {file_num+1}/{amt_files}")
        np.save(f"{target_dir}/GPC_{file[:-4]}.npy", local_gpc_systems)


def compute_descriptors(directory, target_dir):
    """Computes SHOT-descriptors for the MPI-FAUST registration files."""

    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]
    for f in tqdm(file_list[:1]):
        mesh = trimesh.load_mesh(f"{directory}/{f}")
        descr = pyshot.get_descriptors(
            np.array(mesh.vertices),
            np.array(mesh.faces, dtype=np.int64),
            radius=100,
            local_rf_radius=.1,
            min_neighbors=3,
            n_bins=32
        )
        descriptor_filename = f"{target_dir}/SHOT_{f[:-4]}.npy"
        np.save(descriptor_filename, descr)


def preprocess(path, gpc_max_radius, kernel_size, use_c, eps):
    # Note that this directory will be deleted (recursively!) when preprocessing is finished.
    # Only the similar named zip-file will be kept. The zip-file will be used by the 'MpiFaust'-dataset class.
    TARGET_DIR = "./dataset/MPI_FAUST/preprocessed_faust"
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    print("1/4 Computing shot descriptors for registration files.")
    compute_descriptors(path, TARGET_DIR)

    print("2/4 Computing local GPC-systems for registration files.")
    compute_gpc_systems(path, TARGET_DIR, gpc_max_radius, eps, use_c)

    print("3/4 Computing Barycentric coordinates for registration files.")
    compute_barycentric_coordinates(
        path,
        TARGET_DIR,
        k_radius=gpc_max_radius-0.01,
        k_radial_coords=kernel_size[0],
        k_angular_coords=kernel_size[1]
    )

    print("4/4 Zip data and remove target directory to save space.")
    shutil.make_archive(TARGET_DIR, "zip", TARGET_DIR)
    shutil.rmtree(TARGET_DIR)

    print("Preprocessing finished.")


if __name__ == "__main__":
    FAUST_REGISTRATIONS = "/home/andreas/Uni/Masterarbeit/MPI-FAUST/training/registrations"
    GPC_MAX_RADIUS = 0.05
    KERNEL_SIZE = (2, 4)
    USE_C = True
    EPS = 0.000001
    preprocess(FAUST_REGISTRATIONS, GPC_MAX_RADIUS, KERNEL_SIZE, USE_C, EPS)
