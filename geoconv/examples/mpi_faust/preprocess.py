from geoconv.utils.misc import shuffle_mesh_vertices
from geoconv.preprocessing.discrete_gpc import compute_gpc_systems
from geoconv.preprocessing.barycentric_coords import barycentric_coordinates

import os
import tqdm
import trimesh
import numpy as np
import shutil


def create_datasets(directory,
                    target_dir,
                    reference_mesh,
                    n_radial_set,
                    n_angular_set,
                    radius_set):
    """Creates a dataset from triangle meshes.

    Parameters
    ----------
    directory: str
        The directory of the triangle meshes
    target_dir: str
        The directory where to store the preprocessed meshes
    reference_mesh: str
        The reference mesh for the ground-truth computation
    n_radial_set: list
        A list of amounts of radial coordinates for the kernel to apply
    n_angular_set: list
        A list of amounts of angular coordinates for the kernel to apply
    radius_set: list
        The radius of the kernel
    """
    for n_radial in n_radial_set:
        for n_angular in n_angular_set:
            for radius in radius_set:
                radius_str = f"{radius}"[2:]  # without everything in front of the comma
                target_dir_extended = f"{target_dir}_{n_radial}_{n_angular}_{radius_str}"
                preprocess(
                    directory, target_dir_extended, reference_mesh, n_radial, n_angular, radius
                )


def preprocess(directory, target_dir, reference_mesh, n_radial, n_angular, radius):
    """For each mesh process hks-descriptors, barycentric coordinates and ground-truth values.

    Parameters
    ----------
    directory: str
        The directory of the triangle meshes
    target_dir: str
        The directory where to store the preprocessed meshes
    reference_mesh: str
        The reference mesh for the ground-truth computation
    n_radial: int
        The amount of radial coordinates for the kernel to apply
    n_angular: int
        The amount of angular coordinates for the kernel to apply
    radius: float
        The radius of the kernel
    """

    # Prepare mesh target directory
    if os.path.exists(f"{target_dir}.zip"):
        print(f"{target_dir}.zip already exists!")
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Prepare mesh files
    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]

    # Load reference mesh
    reference_mesh = trimesh.load_mesh(reference_mesh)
    ground_truth = reference_mesh.vertices[:, 0].argsort()
    reference_mesh.vertices = reference_mesh.vertices[ground_truth]
    print(f"CHOSEN KERNEL SIZE: radius = {radius}; n_radial = {n_radial}; n_angular = {n_angular}")

    with tqdm.tqdm(total=len(file_list)) as pbar:
        for file_no, file in enumerate(file_list):
            # Load query mesh
            mesh = trimesh.load_mesh(f"{directory}/{file}")

            # Shuffle vertices of query mesh (otherwise the ground truth matrix equals unit matrix)
            mesh, shuffled_node_indices = shuffle_mesh_vertices(mesh)

            # Names for files that contain preprocessed information
            hks_name = f"{target_dir}/HKS_{file[:-4]}.npy"
            bc_name = f"{target_dir}/BC_{file[:-4]}.npy"
            gt_name = f"{target_dir}/GT_{file[:-4]}.npy"

            if not os.path.exists(hks_name) or not os.path.exists(bc_name):
                # TODO: Store heat-kernel-signature of the mesh
                extractor = SignatureExtractor(mesh, 100, approx='beltrami')
                heat_fs = extractor.signatures(64, 'heat')
                np.save(hks_name, heat_fs)

                # Store ground truth
                np.save(gt_name, shuffled_node_indices.astype(np.int16))

                if not os.path.exists(bc_name):

                    # Compute local GPC-systems
                    pbar.set_postfix({"Step": "Compute local GPC-systems"})
                    local_gpc_systems = compute_gpc_systems(
                        mesh,
                        u_max=radius,
                        eps=.000001,
                        use_c=True,
                        tqdm_msg=f"File {file_no} - Compute local GPC-systems"
                    ).astype(np.float64)

                    # Compute Barycentric coordinates
                    pbar.set_postfix({"Step": "Compute Barycentric coordinates"})
                    bary_coords = barycentric_coordinates(
                        mesh, local_gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=radius - 0.005
                    )
                    np.save(bc_name, bary_coords)

            pbar.update(1)

    shutil.make_archive(target_dir, "zip", target_dir)
    shutil.rmtree(target_dir)
    print("Preprocessing finished.")
