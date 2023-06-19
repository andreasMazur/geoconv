from geoconv.preprocessing.barycentric_coordinates import create_kernel_matrix, polar_to_cart, compute_barycentric_coordinates
from geoconv.preprocessing.discrete_gpc import compute_gpc_systems
from geoconv.utils.visualization import draw_gpc_triangles, gpc_on_mesh
from geoconv.utils.misc import reconstruct_kernel, find_smallest_radius

from pathlib import Path

import open3d as o3d
import trimesh
import numpy as np


def load_bunny(target_triangles_amount=6000):
    """Loads and simplifies the Stanford bunny

    Download the Stanford bunny from here:
    https://github.com/alecjacobson/common-3d-test-models/blob/master/data/stanford-bunny.zip

    Unzip the .zip-file and move the 'bun_zipper.ply'-file into the folder where this demo-file
    is saved.

    Parameters
    ----------
    target_triangles_amount: int
        The amount of target triangles of the stanford bunny triangle mesh. Watch out, the target
        amount of triangles is not guaranteed!

    Returns
    -------
    trimesh.Trimesh:
        The Stanford bunny triangle mesh.
    """

    b = o3d.io.read_triangle_mesh(PATH_TO_STANFORD_BUNNY)
    b.compute_vertex_normals()
    b = b.simplify_quadric_decimation(target_number_of_triangles=target_triangles_amount)
    b = trimesh.Trimesh(vertices=b.vertices, faces=b.triangles)

    return b


def preprocess(recompute_gpc=False, recompute_bc=False):
    # Load the stanford bunny in '.ply'-format as a trimesh-object. Set your path 'PATH_TO_STANFORD_BUNNY' correctly in
    # this script below.
    bunny = load_bunny()

    # Find the smallest distance from a center vertex in the bunny mesh to one of its one-hop neighbors.
    # Why? Because using a distance smaller than that distance will cause the GPC-computation-algorithm
    # to return GPC-systems which exceed the given max-radius.
    u_max = find_smallest_radius(bunny)

    # Set the maximal radial distance for the GPC-systems. If selected large enough, no mesh-vertex will
    # have a radial coordinate larger than 'u_max'.
    u_max = u_max + u_max * .1

    # Compute and store the GPC-systems for the bunny mesh.
    bunny_path_gpc = "bunny_gpc_systems_gpc.npy"
    if not Path(bunny_path_gpc).is_file() or recompute_gpc:
        gpc_systems = compute_gpc_systems(bunny, u_max=u_max, tqdm_msg="Computing GPC-systems", use_c=True)
        np.save(bunny_path_gpc, gpc_systems)
    else:
        gpc_systems = np.load(bunny_path_gpc)

    # Select the kernel radius to be 3/4-th of the GPC-systems max-radius to increase the likelihood of
    # the kernel vertices to fall into the GPC-system and not exceed its bounds.
    kernel_radius = u_max * 0.75

    # Compute the barycentric coordinates for the kernel in the computed GPC-systems.
    bunny_path_bc = "bunny_gpc_systems_bc.npy"
    if not Path(bunny_path_bc).is_file() or recompute_bc:
        bc = compute_barycentric_coordinates(
            bunny, gpc_systems, n_radial=N_RADIAL, n_angular=N_ANGULAR, radius=kernel_radius, verbose=True
        )
        np.save(bunny_path_bc, bc)
    else:
        bc = np.load(bunny_path_bc)

    # Visualization of the GPC-systems and the barycentric coordinates.
    for gpc_system_idx in range(bc.shape[0]):
        # Original kernel
        kernel_matrix = create_kernel_matrix(n_radial=N_RADIAL, n_angular=N_ANGULAR, radius=kernel_radius)
        for rc in range(kernel_matrix.shape[0]):
            for ac in range(kernel_matrix.shape[1]):
                kernel_matrix[rc, ac] = polar_to_cart(kernel_matrix[rc, ac, 1], kernel_matrix[rc, ac, 0])

        # Kernel depicted via barycentric coordinates
        rk = reconstruct_kernel(gpc_systems[gpc_system_idx], bc[gpc_system_idx])

        for radial_coordinate in range(bc.shape[1]):
            for angular_coordinate in range(bc.shape[2]):
                print(
                    f"Location: {(gpc_system_idx, radial_coordinate, angular_coordinate)}: "
                    f"Vertices: {bc[gpc_system_idx, radial_coordinate, angular_coordinate, :, 0]} "
                    f"B.c: {bc[gpc_system_idx, radial_coordinate, angular_coordinate, :, 1]} "
                    f"-> Caused: {rk[radial_coordinate, angular_coordinate]}"
                )
        print("==========================================================================")

        # Draw GPC-system on the mesh (center_vertex, radial_coordinates, angular_coordinates, object_mesh)
        gpc_on_mesh(
            gpc_system_idx,
            gpc_systems[gpc_system_idx, :, 0],
            gpc_systems[gpc_system_idx, :, 1],
            bunny
        )

        # Original/Set kernel vertices
        draw_gpc_triangles(
            bunny,
            gpc_system_idx,
            u_max=u_max,
            kernel_matrix=kernel_matrix,
            plot=False,
            title="GPC-system in 2D with kernel vertices"
        )

        # With barycentric coordinates and kernel vertices reconstructed kernel vertices
        draw_gpc_triangles(
            bunny,
            gpc_system_idx,
            u_max=u_max,
            kernel_matrix=rk,
            scatter_color="blue",
            title="GPC-system in 2D with reconstructed kernel vertices"
        )


if __name__ == "__main__":

    # Path to the stanford bunny
    PATH_TO_STANFORD_BUNNY = "bun_zipper.ply"

    # Kernel configuration
    N_RADIAL = 4
    N_ANGULAR = 4

    # Start preprocess
    preprocess()
