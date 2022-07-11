from dataset.MPI_FAUST.tf_dataset import faust_generator
from preprocessing.barycentric_coords import barycentric_coordinates, create_kernel_matrix

import numpy as np


if __name__ == "__main__":
    gen = faust_generator("/home/andreas/PycharmProjects/Masterarbeit/dataset/MPI_FAUST/preprocessed_registrations.zip")
    (_, _, local_gpc_systems, object_mesh), _ = next(gen)

    kernel = create_kernel_matrix(n_radial=2, n_angular=4, radius=.04)

    W_ijkl = barycentric_coordinates(local_gpc_systems, kernel, object_mesh)
    np.save("./test_bary_coords", W_ijkl)
