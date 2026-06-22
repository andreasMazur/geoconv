from geoconv.preprocessing.atlas import Atlas, load_atlas
from geoconv_examples.modelnet.preprocess import save_atlas

import os
import trimesh
import numpy as np


def preprocess_faust(registration_dir,
                     output_path,
                     template_resolutions,
                     max_chart_radius,
                     method="hdm",
                     normalization_method="hdm",
                     processes=1):
    """Preprocesses the FAUST dataset: computes the atlases for all FAUST shapes and barycentric coordinates.

    Parameters
    ----------
    registration_dir: str
        The path to where the raw FAUST registrations are stored.
    output_path: str
        The path to where the preprocessed FAUST meshes are stored.
    template_resolutions: list
        A list of tuples, each describing a template resolution. Those are the ones for which barycentric coordinates
        are computed.
    max_chart_radius: float
        The maximum allowed chart radius for all surface charts.
    method: str
        The used method for computing local charts.
    normalization_method: str
        The used method for computing the geodesic diameter for shape normalization.
    processes: int
        The amount of concurrent processes to use for computing local charts and barycentric coordinates.
    """
    # 1.) Prepare path to registration directory
    registrations = [f for f in os.listdir(registration_dir) if f.endswith(".ply")]
    registrations.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # 2.) Prepare output directory
    os.makedirs(output_path, exist_ok=True)

    # 3.) Compute local charts
    gpc_system_radii = []
    for ply_filename in registrations:
        mesh_save_path = f"{output_path}/{ply_filename.split('.')[0]}.hdf5"

        # Load the mesh
        mesh_filepath = f"{registration_dir}/{ply_filename}"
        print(f"[GPC system] Loading: '{mesh_filepath}'")
        mesh = trimesh.load(mesh_filepath)

        # Compute mesh permutation
        permutation = np.arange(mesh.vertices.shape[0]).astype(np.int32)
        np.random.shuffle(permutation)
        inverse_permutation = np.zeros(mesh.vertices.shape[0]).astype(np.int32)
        for idx, perm_idx in enumerate(permutation):
            inverse_permutation[perm_idx] = int(idx)
        mesh.vertices = mesh.vertices[permutation]
        mesh.faces = inverse_permutation[mesh.faces]

        # Compute the atlas
        atlas = Atlas(
            triangle_mesh=mesh,
            max_radius=max_chart_radius,
            method=method,
            normalization_method=normalization_method,
            processes=processes
        )
        atlas.store_array({"ground_truth": permutation})
        save_atlas(atlas, mesh_save_path)

        # Remember chart radii for BC computation
        gpc_system_radii.extend(atlas.chart_radii.tolist())

    # 4.) Compute barycentric coordinates
    for ply_filename in registrations:
        # Load atlas
        mesh_save_path = f"{output_path}/{ply_filename.split('.')[0]}.hdf5"
        atlas = load_atlas(mesh_save_path)

        # Compute BC for all template resolutions
        for template_resolution in template_resolutions:
            n_radial, n_angular = template_resolution
            # Compute BC for all template radii
            for template_radius in [np.median(gpc_system_radii)]:
                print(
                    f"[BC computation] Calculating BC "
                    f"'{n_radial, n_angular, template_radius}' for '{ply_filename}']"
                )
                atlas.determine_barycentric_coordinates(
                    n_radial=n_radial,
                    n_angular=n_angular,
                    radius=template_radius
                )

        # Save atlas
        atlas.save_training_data(mesh_save_path[:-5])

        # Cleanup old atlas file
        os.remove(mesh_save_path)
