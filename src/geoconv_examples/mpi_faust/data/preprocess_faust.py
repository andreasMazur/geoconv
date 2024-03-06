from geoconv.preprocessing.barycentric_coordinates import compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.misc import shuffle_mesh_vertices, normalize_mesh, find_largest_one_hop_dist

from pathlib import Path

import pyshot
import os
import trimesh
import numpy as np
import shutil
import json


def get_file_number(file_name):
    """Extracts the file number contained in the file name

    Parameters
    ----------
    file_name: str
        The file name

    Returns
    -------
    int:
        The file number contained in the file name
    """
    # file_name.split(".")[0] -> Without file ending
    for elem in file_name.split(".")[0].split("_"):
        if elem.isdigit():
            return int(elem)
    raise RuntimeError(f"Filename '{file_name}' has no digit.")


def preprocess_faust(n_radial,
                     n_angular,
                     target_dir,
                     registration_path,
                     shot=True,
                     geodesic_diameters_path="",
                     precomputed_gpc_radius=-1.,
                     processes=1,
                     add_noise=False):
    """Preprocesses the FAUST-data set

    The FAUST-data set has to be downloaded from: https://faust-leaderboard.is.tuebingen.mpg.de/

    It was published in:
    > [FAUST: Dataset and evaluation for 3D mesh registration.]
    (https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Bogo_FAUST_Dataset_and_2014_CVPR_paper.html)
    > Bogo, Federica, et al.

    Parameters
    ----------
    n_radial: int
        The amount of radial coordinates for the kernel in your geodesic convolution.
    n_angular: int
        The amount of angular coordinates for the kernel in your geodesic convolution.
    target_dir: str
        The path to the directory in which the preprocessing results will be stored. At the end of the preprocessing
        procedure this directory will be replaced with a zip-compressed archive.
    registration_path: str
        The path to the training-registration meshes of the FAUST-data set.
    shot: bool
        Whether to compute SHOT-descriptors as mesh signal.
    geodesic_diameters_path: str
        Path, which points to *.npy file, that contains the geodesic diameters for all the meshes in the dataset.
    precomputed_gpc_radius: float
        The GPC-system radius to use for GPC-system computation. If not provided, the script will calculate it.
    processes: int
        The amount of concurrent processes that compute GPC-systems.
    add_noise: bool
        Adds Gaussian noise to the mesh data.

    Returns
    -------
    float:
        The used kernel radius.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    ##################
    # Load mesh-paths
    ##################
    paths_reg_meshes = os.listdir(registration_path)
    paths_reg_meshes.sort(key=get_file_number)
    paths_reg_meshes = [f for f in paths_reg_meshes if f[-4:] != ".png"]

    # Create temp dir for normalized meshes
    temp_dir = "./temp_meshes"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Check whether geodesic diameters have already been computed
    if Path(geodesic_diameters_path).is_file():
        geodesic_diameters = np.load(geodesic_diameters_path)
        print(f"\n\nLoaded geodesic diameters from: '{geodesic_diameters_path}'\n\n")
    else:
        geodesic_diameters = np.full(len(paths_reg_meshes), -1.)

    # Determine GPC-system-radius
    gpc_radius = .0
    for file_idx in range(len(paths_reg_meshes)):
        # Define file names for normalized vertices and faces (=temp.-meshes)
        reg_file_name = f"{registration_path}/{paths_reg_meshes[file_idx]}"
        normalized_v_name = f"{temp_dir}/vertices_{file_idx}.npy"
        normalized_f_name = f"{temp_dir}/faces_{file_idx}.npy"
        reg_mesh = trimesh.load_mesh(reg_file_name)

        # Check whether normalized meshes already exist
        if not (Path(normalized_v_name).is_file() and Path(normalized_f_name).is_file()):
            # Center and normalize mesh to unit geodesic diameter
            if geodesic_diameters[file_idx] == -1.:
                reg_mesh, geodesic_diameter = normalize_mesh(reg_mesh)
                geodesic_diameters[file_idx] = geodesic_diameter
            else:
                reg_mesh, geodesic_diameter = normalize_mesh(reg_mesh, geodesic_diameters[file_idx])

            # Add noise
            if add_noise:
                reg_mesh.vertices = reg_mesh.vertices + np.random.normal(size=(6890, 3), loc=0, scale=0.0005)

            # Save normalized mesh
            np.save(normalized_v_name, np.asarray(reg_mesh.vertices))
            np.save(normalized_f_name, np.asarray(reg_mesh.faces))
        else:
            vertices = np.load(normalized_v_name)
            faces = np.load(normalized_f_name)
            reg_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            print(f"Found temp-files:\n{normalized_v_name}\n{normalized_f_name}\nSkipping to next normalization..")

        if precomputed_gpc_radius == -1.:
            new_candidate = find_largest_one_hop_dist(reg_mesh)
            gpc_radius = new_candidate if new_candidate > gpc_radius else gpc_radius
    if precomputed_gpc_radius > 0:
        gpc_radius = precomputed_gpc_radius
    else:
        gpc_radius = gpc_radius + gpc_radius * .1
    kernel_radius = gpc_radius * 0.75
    print(f"GPC-system radius: {gpc_radius:.3f} | Kernel radius: {kernel_radius:.3f}")

    # Save computed geodesic diameters
    if not Path(geodesic_diameters_path).is_file():
        geodesic_diameters_name = f"{target_dir}/geodesic_diameters.npy"
        np.save(geodesic_diameters_name, geodesic_diameters)

    # Log GPC-system radius and kernel radius
    properties_file = open(f"{target_dir}/gpc_kernel_properties.json", "w")
    json.dump({"gpc_system_radius": gpc_radius, "kernel_radius": kernel_radius}, properties_file, indent=4)
    properties_file.close()

    for file_idx in range(len(paths_reg_meshes)):
        # Define file names
        bc_name = f"{target_dir}/BC_{paths_reg_meshes[file_idx][:-4]}.npy"
        gt_name = f"{target_dir}/GT_{paths_reg_meshes[file_idx][:-4]}.npy"
        signal_name = f"{target_dir}/SIGNAL_{paths_reg_meshes[file_idx][:-4]}.npy"

        # Load normalized mesh
        vertices = np.load(f"{temp_dir}/vertices_{file_idx}.npy")
        faces = np.load(f"{temp_dir}/faces_{file_idx}.npy")
        reg_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Check whether preprocessed files already exist
        if not (Path(bc_name).is_file() and Path(gt_name).is_file() and Path(signal_name).is_file()):
            #######################################################
            # Shuffle vertices of query mesh and save ground truth
            #######################################################
            reg_mesh, _, ground_truth = shuffle_mesh_vertices(reg_mesh)
            np.save(gt_name, ground_truth)

            ####################
            # Store mesh signal
            ####################
            if shot:
                radius = find_largest_one_hop_dist(reg_mesh) * 2.5
                shot_descrs = pyshot.get_descriptors(
                    reg_mesh.vertices,
                    reg_mesh.faces,
                    radius=radius,
                    local_rf_radius=radius,
                    min_neighbors=10,
                    n_bins=16,
                    double_volumes_sectors=True,
                    use_interpolation=True,
                    use_normalization=True
                )
                np.save(signal_name, shot_descrs)
            else:
                np.save(signal_name, np.asarray(reg_mesh.vertices))

            ############################
            # Compute local GPC-systems
            ############################
            gpc_systems = GPCSystemGroup(reg_mesh, processes=processes)
            gpc_systems.compute(u_max=gpc_radius)

            ##################################
            # Compute Barycentric coordinates
            ##################################
            bary_coords = compute_barycentric_coordinates(
                gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=kernel_radius
            )
            np.save(bc_name, bary_coords)
        else:
            print(f"Found temp-files:\n{bc_name}\n{gt_name}\n{signal_name}\nSkipping to next temp.-mesh..")

    shutil.rmtree(temp_dir)
    shutil.make_archive(target_dir, "zip", target_dir)
    shutil.rmtree(target_dir)
    print("Preprocessing finished.")

    return kernel_radius
