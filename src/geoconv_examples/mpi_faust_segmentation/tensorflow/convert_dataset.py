from geoconv_examples.mpi_faust.tensorflow.faust_data_set import faust_generator
from geoconv_examples.mpi_faust_segmentation.data.segment_meshes import compute_seg_labels

import os
import numpy as np
import shutil
import sys
import pandas as pd


LOADING_CHARS = ["|", "/", "-", "\\"]


def save_mesh_file(idx, signal, bc, gt, coord, new_dataset_path):
    file_number = "".join(["0" for _ in range(3 - len(idx))] + [idx])
    np.save(f"{new_dataset_path}/SIGNAL_tr_reg_{file_number}.npy", signal)
    np.save(f"{new_dataset_path}/BC_tr_reg_{file_number}.npy", bc)
    np.save(f"{new_dataset_path}/GT_tr_reg_{file_number}.npy", gt)
    np.save(f"{new_dataset_path}/COORD_tr_reg_{file_number}.npy", coord)


def convert_dataset(registration_path, old_dataset_path, new_dataset_path):
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    # Compute vertex-to-segment relation
    vertex_segments_relation = compute_seg_labels(registration_path, verbose=False)

    # Load dataset
    dataset = faust_generator(
        path_to_zip=old_dataset_path,
        set_type=3,
        only_signal=False,
        return_coordinates=True
    )

    for idx, ((signal, bc, coord), gt) in enumerate(dataset):
        sys.stdout.write(f"\rTranslating ground truth labels.. {LOADING_CHARS[idx % len(LOADING_CHARS)]}")
        signal, bc, coord, gt = np.array(signal), np.array(bc), np.array(coord), np.array(gt)
        gt_mesh_segmentation = vertex_segments_relation[gt]
        save_mesh_file(f"{idx}", signal, bc, gt_mesh_segmentation, coord, new_dataset_path)

    print("\nCompress converted dataset...")
    shutil.make_archive(new_dataset_path, "zip", new_dataset_path)
    shutil.rmtree(new_dataset_path)
    print("Converting finished.")


def convert_dataset_deepview(csv_path, old_dataset_path, new_dataset_path):
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    # Compute vertex-to-segment relation
    csv = pd.read_csv(csv_path).to_numpy()

    # 1. column: shape index
    # 2. column: vertex index
    # 3. column: corrected class label
    csv = csv[:, [0, 1, 2]].astype(np.int32)

    # Load dataset
    dataset = faust_generator(
        path_to_zip=old_dataset_path,
        set_type=3,
        only_signal=False,
        return_coordinates=True
    )

    for idx, ((signal, bc, coord), gt) in enumerate(dataset):
        sys.stdout.write(f"\rTranslating ground truth labels.. {LOADING_CHARS[idx % len(LOADING_CHARS)]}")

        signal, bc, coord, gt = np.array(signal), np.array(bc), np.array(coord), np.array(gt)

        mesh_corrections = csv[csv[:, 0] == idx][::-1]  # Use last correction for mesh with index 'idx'
        _, array_indices = np.unique(mesh_corrections[:, 1], return_index=True)
        unique_mesh_corrections = mesh_corrections[array_indices]

        # unique_mesh_corrections[:, 1]: vertex indices to correct
        # unique_mesh_corrections[:, 2]: corrected segmentation labels
        gt[unique_mesh_corrections[:, 1]] = unique_mesh_corrections[:, 2]

        # Smooth labels - Vertex takes most prominent neighbor label
        gt = np.array([smooth_label(current_coord, gt, coord) for current_coord in coord])

        save_mesh_file(f"{idx}", signal, bc, gt, coord, new_dataset_path)

    print("\nCompress converted dataset...")
    shutil.make_archive(new_dataset_path, "zip", new_dataset_path)
    shutil.rmtree(new_dataset_path)
    print("Converting finished.")


def smooth_label(current_coord, gt, coord):
    # TODO: Use geodesic distances
    distances = np.linalg.norm(coord - current_coord, axis=1)
    indices = np.argsort(distances)
    classes, occurrences = np.unique(gt[indices[:10]], return_counts=True)
    return classes[occurrences.argmax()]


def coord_to_signal(old_dataset_path, new_dataset_path):
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    dataset = faust_generator(
        path_to_zip=old_dataset_path,
        set_type=3,
        only_signal=False,
        return_coordinates=True
    )

    for idx, ((signal, bc, coord), gt) in enumerate(dataset):
        sys.stdout.write(f"\rTranslating ground truth labels.. {LOADING_CHARS[idx % len(LOADING_CHARS)]}")
        signal, bc, coord, gt = np.array(signal), np.array(bc), np.array(coord), np.array(gt)
        save_mesh_file(f"{idx}", coord, bc, gt, coord, new_dataset_path)

    print("\nCompress converted dataset...")
    shutil.make_archive(new_dataset_path, "zip", new_dataset_path)
    shutil.rmtree(new_dataset_path)
    print("Converting finished.")


def imcnn_new_dataset(imcnn, old_dataset_path, new_dataset_path):
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    dataset = faust_generator(
        path_to_zip=old_dataset_path,
        set_type=3,
        only_signal=False,
        return_coordinates=True
    )

    for idx, ((signal, bc, coord), gt) in enumerate(dataset):
        sys.stdout.write(f"\rTranslating ground truth labels.. {LOADING_CHARS[idx % len(LOADING_CHARS)]}")
        gt = np.array(imcnn([signal, bc])).argmax(axis=-1)
        signal, bc, coord = np.array(signal), np.array(bc), np.array(coord)
        save_mesh_file(f"{idx}", signal, bc, gt, coord, new_dataset_path)

    print("\nCompress converted dataset...")
    shutil.make_archive(new_dataset_path, "zip", new_dataset_path)
    shutil.rmtree(new_dataset_path)
    print("Converting finished.")
