from barycentric_coordinates.barycentric_coords import create_kernel_matrix, barycentric_coordinates
from geodesic_polar_coordinates.discrete_gpc import discrete_gpc

from scipy import sparse

import trimesh
import os


def prepare_triangular_mesh(path, kernel_size, use_c=False):
    """Computes the local GPC-systems and barycentric coordinates for a given object mesh.

    **Input**

    - `path`: The path to the '*.ply'-file.
    - `kernel_size`: A triple `(m, n, r)` where `m` is the amount of radial- and `n` the amount of angular coordinates.
      The value `r` sets the radius of the kernel.
    - `use_c`: Flag that tells whether to use the C-extension module. If set to `False` a slower python equivalent
      will be used. Setting it to `True` requires to install the C-extension.

    **Output**

    - The local GPC-systems in an array of size (#nodes, #nodes, 2) [radial, angular]. Row `i` corresponds
      to the local GPC-system centered in vertex `i`. Column `j` is the coordinate w.r.t. center vertex `i`.
    - The Barycentric coordinates w.r.t. the `kernel_size` and each local GPC-system in an array `E` of size
      `(#vertices, #radial, #angular, #vertices)`. `E[i,j,k,l]` contains the barycentric weight of vertex `l` in the
      local GPC-system of vertex `i` for the kernel vertex `(j, k)`.

    """

    object_mesh = trimesh.load_mesh(path)
    GPC = discrete_gpc(object_mesh, u_max=1.5*kernel_size[2], eps=0.000001, use_c=use_c)
    kernel = create_kernel_matrix(n_radial=kernel_size[0], n_angular=kernel_size[1], radius=kernel_size[2])
    E = barycentric_coordinates(GPC, kernel, object_mesh)

    return GPC, E


def prepare_dataset(dir_path, kernel_size, use_c=False):
    """Computes the local GPC-systems and barycentric coordinates for an entire data set of object meshes.

    **Input**

    - `path`: The path to the directory containing all '*.ply'-file of the data set.
    - `kernel_size`: A triple `(m, n, r)` where `m` is the amount of radial- and `n` the amount of angular coordinates.
      The value `r` sets the radius of the kernel.
    - `use_c`: Flag that tells whether to use the C-extension module. If set to `False` a slower python equivalent
      will be used. Setting it to `True` requires to install the C-extension.

    **Output**

    - The Barycentric coordinates stored at f'{dir_path}/pre_processing_results'.

    """

    file_list = os.listdir(dir_path)
    file_list.sort()

    saving_dir = f"{dir_path}/pre_processing_results"
    os.mkdir(saving_dir)
    for file in file_list:
        if file[-4:] == ".ply" or file[-4:] == ".PLY":
            _, E = prepare_triangular_mesh(f"{dir_path}/{file}", kernel_size, use_c)
            mesh_path = f"{saving_dir}/{file[:-4]}"
            os.mkdir(mesh_path)
            for i in range(E.shape[0]):
                for l in range(E.shape[3]):
                    filename = f"{mesh_path}/{i}{l}_barycentric.npz"
                    # Get Barycentric coordinates of vertex `l` in GPC `i`
                    il_barycentric = sparse.csr_matrix(E[i, :, :, l])
                    # Save sparse matrix
                    sparse.save_npz(filename, il_barycentric)
