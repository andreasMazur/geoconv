from geoconv.preprocessing.atlas import Atlas, load_atlas

import os
import trimesh
import shutil
import numpy as np


def clr_atlas(mesh, mesh_save_path, max_chart_radius, method, normalization_method, processes):
    """Computes, loads or repairs an atlas.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The mesh for which an Atlas should be computed.
    mesh_save_path: str
        The path to where the atlas should be saved.
    max_chart_radius: float
        The maximum radius of any chart within the atlas.
    method: str
        Method used to compute GPC-systems.
    normalization_method: str
        Method used to normalize mesh.
    processes: int
        Amount of concurrent processes to use.

    Returns
    -------
    Atlas:
        The computed, loaded or repaired atlas.
    """
    try:
        if not os.path.isfile(mesh_save_path):
            atlas = Atlas(
                triangle_mesh=mesh,
                max_radius=max_chart_radius,
                method=method,
                normalization_method=normalization_method,
                processes=processes
            )
            os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
            atlas.save(mesh_save_path)
            print(f"[compute, load or repair] Computed and stored atlas at: '{mesh_save_path}'")
        else:
            atlas = load_atlas(mesh_save_path)
            print(f"[compute, load or repair] Atlas loaded from: '{mesh_save_path}'.")
    except KeyError:
        atlas = Atlas(
            triangle_mesh=mesh,
            max_radius=max_chart_radius,
            method=method,
            normalization_method=normalization_method,
            processes=processes
        )
        os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
        atlas.save(mesh_save_path)
        print(f"[compute, load or repair] Atlas saved at: '{mesh_save_path}' has been repaired.")
    return atlas


def preprocess_faust(registration_dir,
                     output_path,
                     template_resolutions=None,
                     method="hdm",
                     normalization_method="hdm",
                     processes=1):
    # 1.) Prepare path to registration directory
    registrations = [f for f in os.listdir(registration_dir) if f.endswith(".ply")]
    registrations.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    for max_chart_radius in [0.05, 0.1, 0.15, 0.2]:
        # 2.) Prepare output directory
        radius_output_path = f"{output_path}_{max_chart_radius}"
        if os.path.isfile(f"{radius_output_path}.zip"):
            print(f"[Preprocessing] Already processed: '{radius_output_path}.zip']. Skipping.")
            continue
        os.makedirs(radius_output_path, exist_ok=True)

        # 3.) Compute local charts
        gpc_system_radii = []
        for ply_filename in registrations[:2]:
            mesh_save_path = f"{radius_output_path}/{ply_filename.split('.')[0]}.hdf5"

            # Load the mesh
            mesh_filepath = f"{registration_dir}/{ply_filename}"
            print(f"[GPC system] Loading: '{mesh_filepath}'")
            mesh = trimesh.load(mesh_filepath)

            # Compute the atlas
            atlas = clr_atlas(
                mesh=mesh,
                mesh_save_path=mesh_save_path,
                max_chart_radius=max_chart_radius,
                method=method,
                normalization_method=normalization_method,
                processes=processes
            )

            # Remember chart radii for BC computation
            gpc_system_radii.extend(atlas.chart_radii.tolist())

        # 4.) Compute barycentric coordinates
        if template_resolutions is None:
            template_resolutions = [(2, 4), (4, 8)]
        for template_radius in [np.min(gpc_system_radii), np.median(gpc_system_radii), np.max(gpc_system_radii)]:
            for (n_radial, n_angular) in template_resolutions:
                for ply_filename in registrations[:2]:
                    mesh_save_path = f"{radius_output_path}/{ply_filename.split('.')[0]}.hdf5"
                    atlas = clr_atlas(
                        mesh=mesh,
                        mesh_save_path=mesh_save_path,
                        max_chart_radius=max_chart_radius,
                        method=method,
                        normalization_method=normalization_method,
                        processes=processes
                    )

                    print(
                        f"[BC computation] Calculating BC "
                        f"'{n_radial, n_angular, template_radius}' for '{ply_filename}']"
                    )
                    atlas.determine_barycentric_coordinates(
                        n_radial=n_radial,
                        n_angular=n_angular,
                        radius=template_radius
                    )
                    atlas.save(mesh_save_path)

        # 5.) Zip dataset
        print("Zipping..")
        shutil.make_archive(base_name=radius_output_path, format="zip", root_dir=radius_output_path)
        shutil.rmtree(radius_output_path)
        print("Done.")
