from geoconv.preprocessing.atlas import Atlas, load_atlas

import os
import trimesh
import shutil
import numpy as np


def load_or_repair(mesh_load_path, mesh_save_path, max_chart_radius, method, normalization_method, processes):
    try:
        # Remember chart radii for BC computation
        atlas = load_atlas(mesh_save_path)
        print(f"[load or repair] Atlas loaded from: '{mesh_save_path}'.")
    except KeyError:
        print(f"[load or repair] Repairing atlas for: '{mesh_load_path}'")
        mesh = trimesh.load(mesh_load_path)
        atlas = Atlas(
            triangle_mesh=mesh,
            max_radius=max_chart_radius,
            method=method,
            normalization_method=normalization_method,
            processes=processes
        )
        atlas.save(mesh_save_path)
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
            continue
        os.makedirs(radius_output_path, exist_ok=True)

        # 3.) Compute local charts
        gpc_system_radii = []
        for ply_filename in registrations:
            mesh_save_path = f"{radius_output_path}/{ply_filename.split('.')[0]}.hdf5"
            if not os.path.isfile(mesh_save_path):
                print(f"[GPC system] Preprocessing '{ply_filename}']")
                mesh = trimesh.load(f"{registration_dir}/{ply_filename}")
                atlas = Atlas(
                    triangle_mesh=mesh,
                    max_radius=max_chart_radius,
                    method=method,
                    normalization_method=normalization_method,
                    processes=processes
                )
                atlas.save(mesh_save_path)

                # Remember chart radii for BC computation
                gpc_system_radii.extend(atlas.chart_radii.tolist())
            else:
                atlas = load_or_repair(
                    mesh_load_path=f"{registration_dir}/{ply_filename}",
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
                for ply_filename in registrations:
                    mesh_save_path = f"{radius_output_path}/{ply_filename.split('.')[0]}.hdf5"
                    atlas = load_or_repair(
                        mesh_load_path=f"{registration_dir}/{ply_filename}",
                        mesh_save_path=mesh_save_path,
                        max_chart_radius=max_chart_radius,
                        method=method,
                        normalization_method=normalization_method,
                        processes=processes
                    )

                    print(f"[BC computation] Calculating BC for '{ply_filename}']")
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
