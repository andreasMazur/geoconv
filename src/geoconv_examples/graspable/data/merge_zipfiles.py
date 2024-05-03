from src.geoconv_examples.mpi_faust.pytorch.faust_data_set import faust_generator

import numpy as np
import os
import shutil
import json


def save_mesh_file(file_name, signal, bc, gt, new_dataset_path):
    np.save(f"{new_dataset_path}/SIGNAL_{file_name}.npy", signal)
    np.save(f"{new_dataset_path}/BC_{file_name}.npy", bc)
    np.save(f"{new_dataset_path}/GT_{file_name}.npy", gt)


if __name__ == "__main__":
    root_path = "to/partial/datasets"

    new_dataset_path = f"{root_path}/combined_dataset"
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    file_name = "dataset_partnet_"
    file_endings = [
        "0_10.zip",
        "10_20.zip",
        "20_30.zip",
        # "30_40.zip",
        # "40_50.zip",
        "50_60.zip",
        "60_70.zip",
        # "70_80.zip",
        "80_90.zip",
        # "90_100.zip"
    ]
    smallest_kernel_radius = np.inf
    for ending in file_endings:
        # Total file path to sub-dataset
        file_path = f"{root_path}/{file_name}{ending}"

        # Load sub-dataset
        dataset = faust_generator(file_path, set_type=3)
        for (signal, bc, fn), gt in dataset:
            save_mesh_file(fn, signal, bc, gt, new_dataset_path)

            zip_file = np.load(file_path, allow_pickle=True)
            # zip_file['gpc_kernel_properties.json']
            x = json.loads(zip_file['gpc_kernel_properties.json'])
            if x["kernel_radius"] < smallest_kernel_radius:
                smallest_kernel_radius = x["kernel_radius"]

    with open(f"{new_dataset_path}/gpc_kernel_properties.json", "w") as f:
        json.dump({"kernel_radius": smallest_kernel_radius}, f, indent=4)

    print("\nCompress converted dataset...")
    shutil.make_archive(new_dataset_path, "zip", new_dataset_path)
    shutil.rmtree(new_dataset_path)
    print("Converting finished.")
