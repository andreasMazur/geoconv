from geoconv_examples.improving_segmentation.data.segment_meshes import compute_seg_labels

import os
import numpy as np
import shutil
import sys
import pandas as pd


LOADING_CHARS = ["|", "/", "-", "\\"]


def save_mesh_file(idx, signal, bc, gt, coord, new_dataset_path):
    """Saves a mesh data to a given folder."""
    file_number = "".join(["0" for _ in range(3 - len(idx))] + [idx])
    np.save(f"{new_dataset_path}/SIGNAL_tr_reg_{file_number}.npy", signal)
    np.save(f"{new_dataset_path}/BC_tr_reg_{file_number}.npy", bc)
    np.save(f"{new_dataset_path}/GT_tr_reg_{file_number}.npy", gt)
    np.save(f"{new_dataset_path}/COORD_tr_reg_{file_number}.npy", coord)


def convert_dataset(old_dataset, registration_path, new_dataset_path, signals_are_coordinates=False):
    """Converts shape correspondence labels to shape segmentation labels using pre-defined bounding boxes for FAUST."""
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    # Compute vertex-to-segment relation
    vertex_segments_relation = compute_seg_labels(registration_path, verbose=False)

    if signals_are_coordinates:
        for idx, ((signal, bc), gt) in enumerate(old_dataset):
            sys.stdout.write(f"\rTranslating ground truth labels.. {LOADING_CHARS[idx % len(LOADING_CHARS)]}")
            signal, bc, gt = np.array(signal), np.array(bc), np.array(gt)
            save_mesh_file(f"{idx}", signal, bc, vertex_segments_relation[gt], signal, new_dataset_path)
    else:
        for idx, ((signal, bc, coord), gt) in enumerate(old_dataset):
            sys.stdout.write(f"\rTranslating ground truth labels.. {LOADING_CHARS[idx % len(LOADING_CHARS)]}")
            signal, bc, coord, gt = np.array(signal), np.array(bc), np.array(coord), np.array(gt)
            save_mesh_file(f"{idx}", signal, bc, vertex_segments_relation[gt], coord, new_dataset_path)

    print("\nCompress converted dataset...")
    shutil.make_archive(new_dataset_path, "zip", new_dataset_path)
    shutil.rmtree(new_dataset_path)
    print("Converting finished.")


def induce_label_correction(csv_array,
                            mesh_idx,
                            mesh_signal,
                            mesh_bc,
                            mesh_coordinates,
                            mesh_gt,
                            new_dataset_path,
                            changes_path=None):
    """Converts original ground truth to corrected deep-view ground truth."""
    # Convert to numpy arrays
    signal, bc, coord, gt = np.array(mesh_signal), np.array(mesh_bc), np.array(mesh_coordinates), np.array(mesh_gt)

    # Filter for current mesh (csv[:, 0] stores mesh indices)
    mesh_corrections = csv_array[csv_array[:, 0] == mesh_idx][::-1]

    # Filter for unique corrections, use last update (mesh_corrections[:, 1] stores vertex indices)
    mesh_corrections = mesh_corrections[np.unique(mesh_corrections[:, 1], return_index=True)[1]]

    # Store changes
    if changes_path is not None:
        np.save(f"{changes_path}/mesh_{mesh_idx}", mesh_corrections)

    # unique_mesh_corrections[:, 1]: vertex indices to correct
    # unique_mesh_corrections[:, 2]: corrected segmentation labels
    gt[mesh_corrections[:, 1]] = mesh_corrections[:, 2]

    save_mesh_file(f"{mesh_idx}", signal, bc, gt, coord, new_dataset_path)


def convert_dataset_deepview(csv_path, old_dataset, new_dataset_path, changes_path=None, signals_are_coordinates=False):
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
            induce_label_correction(csv, idx, signal, bc, signal, gt, new_dataset_path, changes_path)
    else:
        for idx, ((signal, bc, coord), gt) in enumerate(old_dataset):
            sys.stdout.write(f"\rTranslating ground truth labels.. {LOADING_CHARS[idx % len(LOADING_CHARS)]}")
            induce_label_correction(csv, idx, signal, bc, coord, gt, new_dataset_path, changes_path)

    print("\nCompress converted dataset...")
    shutil.make_archive(new_dataset_path, "zip", new_dataset_path)
    shutil.rmtree(new_dataset_path)
    print("Converting finished.")
