from pre_processing.barycentric_coordinates import barycentric_weights, create_kernel_matrix

import numpy as np
import trimesh
import os


if __name__ == "__main__":
    # Load polygonal mesh
    faust_dir = "/home/andreas/Uni/Masterarbeit/MPI-FAUST/training/registrations"
    file_list = os.listdir(faust_dir)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]

    # Choose mesh here
    chosen_file = file_list[0]
    object_mesh = trimesh.load_mesh(f"{faust_dir}/{chosen_file}")

    # np.seterr(all="raise")

    kernel = create_kernel_matrix(n_radial=2, n_angular=4, radius=.04)
    local_gpc_systems = np.load("../GeodesicPolarMap/test_gpc_systems.npy")
    W_ijkl = barycentric_weights(local_gpc_systems, kernel, object_mesh)

    print()
