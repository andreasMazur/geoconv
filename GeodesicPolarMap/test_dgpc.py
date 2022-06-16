from GeodesicPolarMap.discrete_gpc import discrete_gpc

import os
import numpy as np
import trimesh


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

    local_gpc_systems = discrete_gpc(object_mesh, u_max=0.32, eps=0.000001)
    np.save("./test_gpc_systems", local_gpc_systems)

    # TODO: Plot mesh nodes and color them according to their geodesic distance
    # TODO: Plot mesh nodes and color them according to their angular coordinate
