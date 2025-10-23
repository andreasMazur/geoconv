from geoconv.preprocessing.atlas import Atlas

import os
import trimesh
import shutil


def preprocess_faust(registration_dir,
                     output_path,
                     max_chart_radius,
                     n_radial,
                     n_angular,
                     max_temp_radius=None,
                     method="hdm",
                     normalization_method="hdm",
                     processes=1):
    # 1.) Prepare path to registration directory
    registrations = [f for f in os.listdir(registration_dir) if f.endswith(".ply")]
    registrations.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # 2.) Prepare output directory
    os.makedirs(output_path, exist_ok=True)

    # 3.) Preprocess shapes
    for ply_filename in registrations:
        mesh = trimesh.load(f"{registration_dir}/{ply_filename}")
        mesh_save_path = f"{output_path}/{ply_filename.split('.')[0]}.hdf5"
        if not os.path.isfile(mesh_save_path):
            print(f"Currently preprocessing: '{ply_filename}'")
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
            atlas.save(mesh_save_path)
        else:
            print(f"Preprocessed dataset already exists at {mesh_save_path}.")

    # 4.) Zip dataset
    print("Zipping..")
    shutil.make_archive(base_name=output_path, format="zip", root_dir=output_path)
    shutil.rmtree(output_path)
    print("Done.")
