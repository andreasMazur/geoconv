from geoconv_examples.detect_graspable_regions.partnet_grasp.dataset import processed_partnet_grasp_generator

from pathlib import Path

import numpy as np
import pandas as pd
import shutil
import sys
import os


LOADING_CHARS = ["|", "/", "-", "\\"]


def save_mesh_file(idx, signal, bc, gt, coord, new_dataset_path):
    """Saves a mesh partnet_grasp to a given folder."""
    file_number = "".join(["0" for _ in range(3 - len(idx))] + [idx])
    np.save(f"{new_dataset_path}/SIGNAL_tr_reg_{file_number}.npy", signal)
    np.save(f"{new_dataset_path}/BC_tr_reg_{file_number}.npy", bc)
    np.save(f"{new_dataset_path}/GT_tr_reg_{file_number}.npy", gt)
    np.save(f"{new_dataset_path}/COORD_tr_reg_{file_number}.npy", coord)


def induce_label_correction(csv_array,
                            mesh_idx,
                            mesh_signal,
                            mesh_bc,
                            mesh_coordinates,
                            mesh_gt,
                            new_dataset_path,
                            label_changes_path=None):
    """Converts original ground truth to corrected deep-view ground truth."""
    # Convert to numpy arrays
    signal, bc, coord, gt = np.array(mesh_signal), np.array(mesh_bc), np.array(mesh_coordinates), np.array(mesh_gt)

    # Filter for current mesh (csv[:, 0] stores mesh indices)
    mesh_corrections = csv_array[csv_array[:, 0] == mesh_idx][::-1]

    # Filter for unique corrections, use last update (mesh_corrections[:, 1] stores vertex indices)
    mesh_corrections = mesh_corrections[np.unique(mesh_corrections[:, 1], return_index=True)[1]]

    # Store effective changes
    if label_changes_path is not None:
        # Set changes for vertices in 'mesh_corrections'
        was_changed = np.zeros(shape=(signal.shape[0]), dtype=np.int32)
        was_changed[mesh_corrections[:, 1]] = 1

        # Correct ineffective changes
        ineffective_changes = mesh_corrections[mesh_corrections[:, 2] == gt[mesh_corrections[:, 1]], 1]
        was_changed[ineffective_changes] = 0

        np.save(f"{label_changes_path}/mesh_changes_{mesh_idx}", was_changed)

    # mesh_corrections[:, 1]: vertex indices to correct
    # mesh_corrections[:, 2]: corrected segmentation labels
    gt[mesh_corrections[:, 1]] = mesh_corrections[:, 2]

    save_mesh_file(f"{mesh_idx}", signal, bc, gt, coord, new_dataset_path)


def convert_dataset_deepview(csv_path,
                             old_dataset,
                             new_dataset_path,
                             label_changes_path=None,
                             signals_are_coordinates=False):
    """Incorporates proposed changes obtained with DeepView into the shape segmentation dataset."""
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    # Compute vertex-to-segment relation
    csv = pd.read_csv(csv_path).to_numpy()

    # 1. column: shape index
    # 2. column: vertex index
    # 3. column: corrected class label
    csv = csv[:, [0, 1, 2]].astype(np.int32)

    if signals_are_coordinates:
        for idx, ((signal, bc), gt) in enumerate(old_dataset):
            sys.stdout.write(f"\rTranslating ground truth labels.. {LOADING_CHARS[idx % len(LOADING_CHARS)]}")
            induce_label_correction(csv, idx, signal, bc, signal, gt, new_dataset_path, label_changes_path)
    else:
        for idx, ((signal, bc, coord), gt) in enumerate(old_dataset):
            sys.stdout.write(f"\rTranslating ground truth labels.. {LOADING_CHARS[idx % len(LOADING_CHARS)]}")
            induce_label_correction(csv, idx, signal, bc, coord, gt, new_dataset_path, label_changes_path)

    print("\nCompress converted dataset...")
    shutil.make_archive(new_dataset_path, "zip", new_dataset_path)
    shutil.rmtree(new_dataset_path)
    print("Converting finished.")


def convert_partnet(old_data_path, new_data_path, csv_path, label_changes_path=None):
    """Integrates label corrections into the old dataset, thereby creating a new one."""
    # Integrate DeepView-corrected labels into the dataset
    old_data_path = f"{old_data_path}.zip"
    if not Path(new_data_path).is_file():
        old_dataset = processed_partnet_grasp_generator(path_to_zip=old_data_path, set_type=3, )
        convert_dataset_deepview(
            csv_path=csv_path,
            old_dataset=old_dataset,
            new_dataset_path=new_data_path,
            signals_are_coordinates=True,
            label_changes_path=label_changes_path
        )
    else:
        print(f"Found converted dataset file: '{new_data_path}'. Skipping preprocessing.")
