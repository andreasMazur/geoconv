from geoconv.preprocessing.barycentric_coordinates import create_template_matrix, polar_to_cart, compute_barycentric_coordinates
from geoconv.preprocessing.gpc_system_group import GPCSystemGroup
from geoconv.utils.visualization import draw_gpc_triangles, draw_gpc_on_mesh, draw_triangles
from geoconv.utils.misc import reconstruct_template, find_largest_one_hop_dist, normalize_mesh

import open3d as o3d
import trimesh


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
                    processes=1):
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
        The amount of radial coordinates for the template in your geodesic convolution.
    n_angular: int
        The amount of angular coordinates for the template in your geodesic convolution.
    """

    # Load the stanford bunny in '.ply'-format as a trimesh-object. Set your path 'PATH_TO_STANFORD_BUNNY' correctly in
    # this script below.
    bunny = load_bunny(path_to_stanford_bunny)

    # Find the smallest distance from a center vertex in the bunny mesh to one of its one-hop neighbors.
    bunny, _ = normalize_mesh(bunny, geodesic_diameter=0.25270776231631265)
    u_max = find_largest_one_hop_dist(bunny)

    # Set the maximal radial distance for the GPC-systems
    u_max = u_max + u_max * .1

    # Select the template radius to be 3/4-th of the GPC-systems max-radius to increase the likelihood of
    # the template vertices to fall into the GPC-system and not exceed its bounds.
    template_radius = u_max * 0.75
    print(f"GPC-system max.-radius: {u_max} | Template max.-radius: {template_radius}")

    # Compute and store the GPC-systems for the bunny mesh.
    gpc_systems = GPCSystemGroup(bunny, processes=processes)  # processes)
    gpc_systems.compute(u_max=u_max)

    # Compute the barycentric coordinates for the template in the computed GPC-systems.
    bc = compute_barycentric_coordinates(
        gpc_systems, n_radial=n_radial, n_angular=n_angular, radius=template_radius, verbose=True
    )

    ####################################################################
    # Visualization of the GPC-systems and the barycentric coordinates
    ####################################################################
    for gpc_system_idx in range(0, bc.shape[0], 100):
        # Original template
        template_matrix = create_template_matrix(n_radial=n_radial, n_angular=n_angular, radius=template_radius)
        for rc in range(template_matrix.shape[0]):
            for ac in range(template_matrix.shape[1]):
                template_matrix[rc, ac] = polar_to_cart(template_matrix[rc, ac, 1], template_matrix[rc, ac, 0])

        # Template depicted via barycentric coordinates
        reconstructed_template = reconstruct_template(
            gpc_systems.object_mesh_gpc_systems[gpc_system_idx].get_gpc_system(), bc[gpc_system_idx]
        )
        for radial_coordinate in range(bc.shape[1]):
            for angular_coordinate in range(bc.shape[2]):
                print(
                    f"Location: {(gpc_system_idx, radial_coordinate, angular_coordinate)}: "
                    f"Vertices: {bc[gpc_system_idx, radial_coordinate, angular_coordinate, :, 0]} "
                    f"B.c: {bc[gpc_system_idx, radial_coordinate, angular_coordinate, :, 1]} "
                    f"-> Caused: {reconstructed_template[radial_coordinate, angular_coordinate]}"
                )
        print("==========================================================================")

        # Draw GPC-system on the mesh (center_vertex, radial_coordinates, angular_coordinates, object_mesh)
        draw_gpc_on_mesh(
            gpc_system_idx,
            gpc_systems.object_mesh_gpc_systems[gpc_system_idx].radial_coordinates,
            gpc_systems.object_mesh_gpc_systems[gpc_system_idx].angular_coordinates,
            bunny
        )

        # Original/Set template vertices
        draw_gpc_triangles(
            gpc_systems.object_mesh_gpc_systems[gpc_system_idx],
            template_matrix=template_matrix,
            plot=True,
            title="GPC-system",
            save_name="gpc_system.svg"
        )

        # Draw triangles with stored GPC-coordinates and place reconstructed template vertices with the help of the
        # computed barycentric coordinates
        # Translate GPC-systems into cartesian coordinates (for visualization purposes).
        draw_triangles(
            gpc_systems.object_mesh_gpc_systems[gpc_system_idx].get_gpc_triangles(in_cart=True),
            points=reconstructed_template.reshape((-1, 2)),
            title="\"Template Vertices in Patch\"",
            plot=True
        )
