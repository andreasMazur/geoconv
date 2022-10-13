from geoconv.preprocessing.barycentric_coords import barycentric_coordinates
from geoconv.preprocessing.discrete_gpc import compute_gpc_systems, local_gpc
from geoconv.utils.measures import evaluate_kernel_coverage

import os
import tqdm
import numpy as np
import scipy
import shutil
import trimesh
import pyshot

from geoconv.utils.misc import shuffle_mesh_vertices


def search_parameters(faust_dir, new_face_count):
    """Search for good preprocessing parameters

    Here, 'good' means that we want to the kernel to cover a large portion
    of the triangles included in the GPC-system.

    Parameters
    ----------
    faust_dir: str
        The directory to the registration files of the faust dataset
    new_face_count: int
        The amount of faces on which the object meshes will be reduced

    Returns
    -------
    (float, int, int)
        - A radius
        - Amount of radial coordinates
        - Amount of angular coordinates
    """

    file_list = os.listdir(faust_dir)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]
    object_mesh = trimesh.load_mesh(f"{faust_dir}/{file_list[0]}")
    object_mesh = object_mesh.simplify_quadratic_decimation(new_face_count)

    source_points = np.random.randint(0, object_mesh.vertices.shape[0], size=(70,))
    best_parameters = (None, 0)
    for radius in [0.02, 0.03, 0.04, 0.05, 0.06]:
        for n_radial in [2, 3, 4]:
            for n_angular in [5, 6, 7, 8, 9, 10]:
                gpc_systems = []
                for source_point in source_points:
                    r_all, theta_all, _ = local_gpc(
                        source_point, u_max=radius, object_mesh=object_mesh, use_c=True, eps=0.000001
                    )
                    gpc_system = np.stack([r_all, theta_all], axis=-1)
                    gpc_systems.append(gpc_system)
                gpc_systems = np.stack(gpc_systems)
                bary_coords = barycentric_coordinates(
                    object_mesh, gpc_systems, n_radial, n_angular, radius - 0.005, verbose=False
                )

                avg_kernel_coverage = evaluate_kernel_coverage(object_mesh, gpc_systems, bary_coords, verbose=False)
                print(
                    f"{(radius, n_radial, n_angular)} - Average kernel coverage: {avg_kernel_coverage * 100:.2f}%"
                    f" - Currently best: ({best_parameters[0]}, {best_parameters[1] * 100:.2f}%)"
                )
                # Larger kernel require a lot more memory, therefore they should improve the coverage more than just
                # a tiny bit.
                if avg_kernel_coverage > best_parameters[1] + 0.05:
                    best_parameters = ((radius, n_radial, n_angular), avg_kernel_coverage)

    print(
        f"Best parameters found: radius = {best_parameters[0][0]}; "
        f"n_radial = {best_parameters[0][1]}; "
        f"n_angular = {best_parameters[0][2]}"
    )
    return best_parameters[0]


def create_datasets(directory,
                    target_dir,
                    reference_mesh,
                    n_faces_set,
                    n_radial_set,
                    n_angular_set,
                    radius_set):
    for n_faces in n_faces_set:
        for n_radial in n_radial_set:
            for n_angular in n_angular_set:
                for radius in radius_set:
                    radius_str = f"{radius}"[2:]  # without everything in front of the comma
                    target_dir_extended = f"{target_dir}_{n_faces}_{n_radial}_{n_angular}_{radius_str}"
                    preprocess(
                        directory, target_dir_extended, reference_mesh, n_radial, n_angular, radius
                    )


def preprocess(directory, target_dir, reference_mesh, n_radial, n_angular, radius):

    if os.path.exists(f"{target_dir}.zip"):
        print(f"{target_dir}.zip already exists!")
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]

    ######################
    # Load reference mesh
    ######################
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

            coordinates_name = f"{target_dir}/COORDS_{file[:-4]}.npy"
            bc_name = f"{target_dir}/BC_{file[:-4]}.npy"
            gt_name = f"{target_dir}/GT_{file[:-4]}.npy"
            if not os.path.exists(coordinates_name) or not os.path.exists(bc_name):
                # Store coordinates
                np.save(coordinates_name, mesh.vertices)

                # Store ground truth
                np.save(gt_name, shuffled_node_indices.astype(np.int16))

                pbar.set_postfix({"Step": "Compute local GPC-systems"})
                if not os.path.exists(bc_name):
                    ############################
                    # Compute local GPC-systems
                    ############################
                    local_gpc_systems = compute_gpc_systems(
                        mesh,
                        u_max=radius,
                        eps=.000001,
                        use_c=True,
                        tqdm_msg=f"File {file_no} - Compute local GPC-systems"
                    ).astype(np.float64)

                    pbar.set_postfix({"Step": "Compute Barycentric coordinates"})
                    ##################################
                    # Compute Barycentric coordinates
                    ##################################
                    bary_coords = barycentric_coordinates(
                        mesh, local_gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=radius - 0.005
                    )
                    np.save(bc_name, bary_coords)

            pbar.update(1)

    shutil.make_archive(target_dir, "zip", target_dir)
    shutil.rmtree(target_dir)
    print("Preprocessing finished.")
