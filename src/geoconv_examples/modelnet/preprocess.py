from geoconv.preprocessing.atlas import Atlas

import point_cloud_utils as pcu
import zipfile
import os
import trimesh
import io
import shutil


def preprocess_modelnet(zip_path,
                        output_path,
                        max_chart_radius,
                        method,
                        normalization_method,
                        processes,
                        n_radial,
                        n_angular,
                        max_temp_radius):
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

        for mesh_filepath in zip_content:
            mesh_save_path = f"{output_path}/{mesh_filepath.split('.')[0]}.hdf5"
            if not os.path.isfile(mesh_save_path):
                # Load the mesh
                print(f"Currently preprocessing: '{mesh_filepath}'")
                mesh = trimesh.load_mesh(io.BytesIO(zip_file.read(mesh_filepath)), file_type="off")

                # Concat meshes if a scene was loaded
                if type(mesh) == trimesh.scene.Scene:
                    mesh = trimesh.util.concatenate([y for y in mesh.geometry.values()])

                # Repair mesh
                new_vertices, new_faces = pcu.make_mesh_watertight(v=mesh.vertices, f=mesh.faces, resolution=1_000)
                mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

                # Start preprocess
                atlas = Atlas(
                    triangle_mesh=mesh,
                    max_radius=max_chart_radius,
                    method=method,
                    normalization_method=normalization_method,
                    processes=processes
                )
                atlas.determine_barycentric_coordinates(
                    n_radial=n_radial,
                    n_angular=n_angular,
                    radius=atlas.median_chart_radius if max_temp_radius is None else max_temp_radius
                )

                os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
                atlas.save(mesh_save_path)
            else:
                print(f"Preprocessed dataset already exists at {mesh_save_path}.")

    # 4.) Zip dataset
    print("Zipping..")
    output_path = f"{output_path}/ModelNet40"
    shutil.make_archive(base_name=output_path, format="zip", root_dir=output_path)
    shutil.rmtree(output_path)
    print("Done.")
