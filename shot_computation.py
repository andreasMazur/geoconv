from tqdm import tqdm

import os
import trimesh
import pyshot
import numpy as np
import shutil


def compute_descriptors(directory, target_directory, down_sample=None):
    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]
    descriptor_files = []
    for f in tqdm(file_list):
        if down_sample is not None:
            mesh = trimesh.load_mesh(f"{directory}/{f}")
            points, face_indices = trimesh.sample.sample_surface_even(mesh, down_sample)
            mesh = mesh.submesh([face_indices])[0]
        else:
            mesh = trimesh.load_mesh(f"{directory}/{f}")
        descr = pyshot.get_descriptors(
            np.array(mesh.vertices),
            np.array(mesh.faces, dtype=np.int64),
            radius=100,
            local_rf_radius=.0,
            min_neighbors=0,
            n_bins=32
        )

        descriptor_filename = f"{target_directory}/{f[:-4]}.npy"
        descriptor_files.append(descriptor_filename)
        np.save(descriptor_filename, descr)
        os.remove(f"{directory}/{f}")

    shutil.make_archive(target_directory, "zip", target_directory)
    for f in descriptor_files:
        os.remove(f)

if __name__ == "__main__":

    # Create directory structure
    for d in ["registrations", "scans"]:
        os.makedirs(f"SHOT-DESCRIPTORS/training/{d}", exist_ok=True)
    os.makedirs("SHOT-DESCRIPTORS/test/scans", exist_ok=True)

    # Compute descriptors
    # print("Currently working on: 'MPI-FAUST/training/registrations'")
    # compute_descriptors(
    #     "MPI-FAUST/training/registrations", "SHOT-DESCRIPTORS/training/registrations"
    # )
    print("Currently working on: 'MPI-FAUST/training/scans'")
    compute_descriptors("MPI-FAUST/training/scans", "SHOT-DESCRIPTORS/training/scans", 20_000)
    print("Currently working on: 'MPI-FAUST/test/scans'")
    compute_descriptors("MPI-FAUST/test/scans", "SHOT-DESCRIPTORS/test/scans", 20_000)
