from tqdm import tqdm

import os
import open3d as o3d
import pyshot
import numpy as np


def compute_descriptors(directory, target_directory):
    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]
    for f in tqdm(file_list):
        mesh = o3d.io.read_triangle_mesh(f"{directory}/{f}")
        descr = pyshot.get_descriptors(
            np.array(mesh.vertices),
            np.array(mesh.triangles, dtype=np.int64),
            radius=100,
            local_rf_radius=.0,
            min_neighbors=0,
            n_bins=32
        )
        np.save(f"{target_directory}/{f[:-4]}.npy", descr)


if __name__ == "__main__":

    # Create directory structure
    for d in ["registrations", "scans"]:
        os.makedirs(f"SHOT-DESCRIPTORS/training/{d}", exist_ok=True)
    os.makedirs("SHOT-DESCRIPTORS/test/scans", exist_ok=True)

    # Compute descriptors
    print("Currently working on: 'MPI-FAUST/training/scans'")
    compute_descriptors("MPI-FAUST/training/scans", "SHOT-DESCRIPTORS/training/scans")
    print("Currently working on: 'MPI-FAUST/training/registrations'")
    compute_descriptors(
        "MPI-FAUST/training/registrations", "SHOT-DESCRIPTORS/training/registrations"
    )
    print("Currently working on: 'MPI-FAUST/test/scans'")
    compute_descriptors("MPI-FAUST/test/scans", "SHOT-DESCRIPTORS/test/scans")
