from geoconv.preprocessing.barycentric_coords import barycentric_coordinates
from geoconv.preprocessing.discrete_gpc import discrete_gpc, local_gpc
from geoconv.utils.measures import evaluate_kernel_coverage

import os
import tqdm
import numpy as np
import scipy
import shutil
import trimesh
import pyshot


def search_parameters(faust_dir):
    """Search for good preprocessing parameters

    Here, 'good' means that we want to the kernel to cover a large portion
    of the triangles included in the GPC-system.

    Parameters
    ----------
    faust_dir: str
        The directory to the registration files of the faust dataset

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
                bary_coords = barycentric_coordinates(object_mesh, gpc_systems, n_radial, n_angular, radius - 0.005)

                avg_kernel_coverage = evaluate_kernel_coverage(object_mesh, gpc_systems, bary_coords, verbose=False)
                print(f"\n{(radius, n_radial, n_angular)} - Average kernel coverage: {avg_kernel_coverage * 100:.2f}%")
                if avg_kernel_coverage > best_parameters[1] * 1.02:
                    best_parameters = ((radius, n_radial, n_angular), avg_kernel_coverage)

    print(
        f"Best parameters found: radius = {best_parameters[0][0]}; "
        f"n_radial = {best_parameters[0][1]}; "
        f"n_angular = {best_parameters[0][2]}"
    )
    return best_parameters[0]


def preprocess(directory, target_dir, reference_mesh):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]

    ######################
    # Load reference mesh
    ######################
    reference_mesh = trimesh.load_mesh(reference_mesh)

    #####################################
    # Find good preprocessing parameters
    #####################################
    radius, n_radial, n_angular = search_parameters(directory)

    with tqdm.tqdm(total=len(file_list)) as pbar:
        for file_no, file in enumerate(file_list):
            pbar.set_postfix({"Step": "Sub-sample the original meshes"})
            ############
            # Load mesh
            ############
            mesh = trimesh.load_mesh(f"{directory}/{file}")

            pbar.set_postfix({"Step": "Ground-truth computation"})
            ################
            # Shuffle nodes
            ################
            # Otherwise ground-truth matrix is unit-matrix all the time
            shuffled_node_indices = np.arange(mesh.vertices.shape[0])
            np.random.shuffle(shuffled_node_indices)
            object_mesh_vertices = np.copy(mesh.vertices)[shuffled_node_indices]
            object_mesh_faces = np.copy(mesh.faces)
            for face in object_mesh_faces:
                face[0] = np.where(shuffled_node_indices == face[0])[0]
                face[1] = np.where(shuffled_node_indices == face[1])[0]
                face[2] = np.where(shuffled_node_indices == face[2])[0]
            mesh = trimesh.Trimesh(vertices=object_mesh_vertices, faces=object_mesh_faces)

            ##########################
            # Set ground-truth labels
            ##########################
            label_matrix = np.zeros(
                shape=(np.array(mesh.vertices).shape[0], np.array(reference_mesh.vertices).shape[0]), dtype=np.int8
            )
            # For vertex mesh.vertices[i] ground truth is given by shuffled_node_indices[i]
            label_matrix[(np.arange(label_matrix.shape[0]), shuffled_node_indices)] = 1
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
                mesh, u_max=radius, eps=.000001, use_c=True, tqdm_msg=f"File {file_no} - Compute local GPC-systems"
            ).astype(np.float64)

            pbar.set_postfix({"Step": "Compute Barycentric coordinates"})
            ##################################
            # Compute Barycentric coordinates
            ##################################
            bary_coords = barycentric_coordinates(
                mesh, local_gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=radius - 0.005
            )
            np.save(f"{target_dir}/BC_{file[:-4]}.npy", bary_coords)

            pbar.update(1)

    shutil.make_archive(target_dir, "zip", target_dir)
    shutil.rmtree(target_dir)
    print("Preprocessing finished.")
