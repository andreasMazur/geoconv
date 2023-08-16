from geoconv.preprocessing.barycentric_coordinates import create_kernel_matrix, polar_to_cart, \
    compute_barycentric_coordinates, determine_gpc_triangles
from geoconv.preprocessing.discrete_gpc import compute_gpc_systems
from geoconv.utils.visualization import draw_gpc_triangles, draw_gpc_on_mesh, draw_triangles
from geoconv.utils.misc import reconstruct_kernel, find_smallest_radius, gpc_systems_into_cart, normalize_mesh

from pathlib import Path

import open3d as o3d
import trimesh
import numpy as np
import os


def load_bunny(path, target_triangles_amount=6000):
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
    path: str
        If the path to the Stanford bunny .ply-file differs from 'PATH_TO_STANFORD_BUNNY' you can
        set it correctly with the 'path'-argument.

    Returns
    -------
    trimesh.Trimesh:
        The Stanford bunny triangle mesh.
    """

    b = o3d.io.read_triangle_mesh(path)
    b = b.simplify_quadric_decimation(target_number_of_triangles=target_triangles_amount)
    b = trimesh.Trimesh(vertices=b.vertices, faces=b.triangles)

    return b


def preprocess_demo(path_to_stanford_bunny="bun_zipper.ply",
                    n_radial=3,
                    n_angular=8,
                    recompute_gpc=False,
                    recompute_bc=False):
    """Demonstrates and visualizes GeoConv's preprocessing at the hand of the stanford bunny.

    Download the Stanford bunny from here:
    https://github.com/alecjacobson/common-3d-test-models/blob/master/data/stanford-bunny.zip

    Unzip the .zip-file and move the 'bun_zipper.ply'-file into the folder where this demo-file
    is saved.

    Parameters
    ----------
    path_to_stanford_bunny: str
        The path to the 'bun_zipper.ply'-file containing the stanford bunny.
    n_radial: int
        The amount of radial coordinates for the kernel in your geodesic convolution.
    n_angular: int
        The amount of angular coordinates for the kernel in your geodesic convolution.
    recompute_gpc: bool
        Force the function to recompute the GPC-systems even if they were computed and stored before.
    recompute_bc: bool
        Force the function to recompute the barycentric coordinates even if they were computed and stored before.
    """

    # Load the stanford bunny in '.ply'-format as a trimesh-object. Set your path 'PATH_TO_STANFORD_BUNNY' correctly in
    # this script below.
    bunny = load_bunny(path_to_stanford_bunny)

    # Find the smallest distance from a center vertex in the bunny mesh to one of its one-hop neighbors.
    bunny = normalize_mesh(bunny)
    u_max = find_smallest_radius(bunny)

    # Set the maximal radial distance for the GPC-systems
    u_max = u_max + u_max * .1

    # Compute and store the GPC-systems for the bunny mesh.
    path = os.path.dirname(path_to_stanford_bunny)
    bunny_path_gpc = f"{path}/bunny_gpc_systems.npy"
    if not Path(bunny_path_gpc).is_file() or recompute_gpc:
        gpc_systems = compute_gpc_systems(bunny, u_max=u_max, tqdm_msg="Computing GPC-systems", use_c=True)
        np.save(bunny_path_gpc, gpc_systems)
    else:
        gpc_systems = np.load(bunny_path_gpc)

    # Select the kernel radius to be 3/4-th of the GPC-systems max-radius to increase the likelihood of
    # the kernel vertices to fall into the GPC-system and not exceed its bounds.
    kernel_radius = u_max * 0.75
    print(f"GPC-system max.-radius: {u_max} | Kernel max.-radius: {kernel_radius}")

    # Compute the barycentric coordinates for the kernel in the computed GPC-systems.
    bunny_path_bc = f"{path}/bunny_barycentric_coordinates.npy"
    if not Path(bunny_path_bc).is_file() or recompute_bc:
        bc = compute_barycentric_coordinates(
            bunny, gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=kernel_radius, verbose=True
        )
        np.save(bunny_path_bc, bc)
    else:
        bc = np.load(bunny_path_bc)

    ####################################################################
    # Visualization of the GPC-systems and the barycentric coordinates
    ####################################################################

    # Translate GPC-systems into cartesian coordinates (for visualization purposes).
    gpc_systems_cart = gpc_systems_into_cart(gpc_systems)

    for gpc_system_idx in range(bc.shape[0]):
        # Original kernel
        kernel_matrix = create_kernel_matrix(n_radial=n_radial, n_angular=n_angular, radius=kernel_radius)
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
        draw_gpc_on_mesh(
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

        # Draw triangles with stored GPC-coordinates and place reconstructed kernel vertices with the help of the
        # computed barycentric coordinates
        triangles, _ = determine_gpc_triangles(bunny, gpc_systems_cart[gpc_system_idx])
        draw_triangles(
            triangles, points=rk.reshape((-1, 2)), title="Reconstructed GPC-system and kernel vertices"
        )
