from geoconv_examples.mpi_faust.tensorflow.faust_data_set import faust_generator
from geoconv_examples.mpi_faust_segmentation.data.segment_meshes import compute_seg_labels

import os
import numpy as np
import shutil


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
        signal, bc, coord, gt = np.array(signal), np.array(bc), np.array(coord), np.array(gt)
        gt_mesh_segmentation = vertex_segments_relation[gt]

        idx = f"{idx}"
        file_number = "".join(["0" for _ in range(3 - len(idx))] + [idx])
        np.save(f"{new_dataset_path}/SIGNAL_tr_reg_{file_number}.npy", signal)
        np.save(f"{new_dataset_path}/BC_tr_reg_{file_number}.npy", bc)
        np.save(f"{new_dataset_path}/GT_tr_reg_{file_number}.npy", gt_mesh_segmentation)
        np.save(f"{new_dataset_path}/COORD_tr_reg_{file_number}.npy", coord)

    print("Compress converted dataset...")
    shutil.make_archive(new_dataset_path, "zip", new_dataset_path)
    shutil.rmtree(new_dataset_path)
    print("Converting finished.")
