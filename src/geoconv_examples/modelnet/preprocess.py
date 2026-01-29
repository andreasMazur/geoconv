from geoconv.preprocessing.atlas import Atlas
from geoconv_examples.faust.preprocess import load_or_repair

import point_cloud_utils as pcu
import zipfile
import os
import trimesh
import io
import shutil
import numpy as np


def load_modelnet_mesh(zip_file, mesh_filepath):
    # Load the mesh
    mesh = trimesh.load_mesh(io.BytesIO(zip_file.read(mesh_filepath)), file_type="off")

    # Concat meshes if a scene was loaded
    if type(mesh) == trimesh.scene.Scene:
        mesh = trimesh.util.concatenate([y for y in mesh.geometry.values()])

    # Repair mesh
    new_vertices, new_faces = pcu.make_mesh_watertight(v=mesh.vertices, f=mesh.faces, resolution=1_000)
    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)


def preprocess_modelnet(zip_path,
                        output_path,
                        template_resolutions=None,
                        method="hdm",
                        normalization_method="hdm",
                        processes=1):
    """Preprocess ModelNet40 shapes.

    Uses the algorithm of:
    > [Robust watertight manifold surface generation method for shapenet models](https://arxiv.org/abs/1802.01698)
    > Jingwei Huang, Hao Su and Leonidas Guibas
    to make meshes watertight for distance calculations.

    Implementation available at:
    > https://fwilliams.info/point-cloud-utils/
    """
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_content = [f for f in zip_file.namelist() if f.endswith(".off")]
        zip_content.sort()

        for max_chart_radius in [0.05, 0.1, 0.15, 0.2]:
            # 2.) Prepare output directory
            radius_output_path = f"{output_path}_{max_chart_radius}"
            if os.path.isfile(f"{radius_output_path}.zip"):
                print(f"[Preprocessing] Already processed: '{radius_output_path}.zip']. Skipping.")
                continue
            os.makedirs(radius_output_path, exist_ok=True)

            # 3.) Compute local charts
            gpc_system_radii = []
            for mesh_filepath in zip_content:
                mesh_save_path = f"{radius_output_path}/{mesh_filepath.split('.')[0]}.hdf5"

                # Load the mesh
                print(f"[GPC system] Loading: '{mesh_filepath}'")
                mesh = load_modelnet_mesh(zip_file, mesh_filepath)

                if not os.path.isfile(mesh_save_path):
                    print(f"[GPC system] Preprocessing '{mesh_filepath}']")
                    atlas = Atlas(
                        triangle_mesh=mesh,
                        max_radius=max_chart_radius,
                        method=method,
                        normalization_method=normalization_method,
                        processes=processes
                    )
                    os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
                    atlas.save(mesh_save_path)

                    # Remember chart radii for BC computation
                    gpc_system_radii.extend(atlas.chart_radii.tolist())
                else:
                    print(f"[GPC system] '{mesh_save_path}' already exists. Loading to gather chart-radii.")
                    atlas = load_or_repair(
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
                    for mesh_filepath in zip_content:
                        mesh_save_path = f"{radius_output_path}/{mesh_filepath.split('.')[0]}.hdf5"
                        atlas = load_or_repair(
                            mesh=load_modelnet_mesh(zip_file, mesh_filepath),
                            mesh_save_path=mesh_save_path,
                            max_chart_radius=max_chart_radius,
                            method=method,
                            normalization_method=normalization_method,
                            processes=processes
                        )

                        print(
                            f"[BC computation] Calculating BC "
                            f"'{n_radial, n_angular, template_radius}' for '{mesh_filepath}']"
                        )
                        atlas.determine_barycentric_coordinates(
                            n_radial=n_radial,
                            n_angular=n_angular,
                            radius=template_radius
                        )
                        atlas.save(mesh_save_path)

            # 4.) Zip dataset
            print("Zipping..")
            shutil.make_archive(base_name=radius_output_path, format="zip", root_dir=radius_output_path)
            shutil.rmtree(radius_output_path)
            print("Done.")
