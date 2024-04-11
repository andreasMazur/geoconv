from geoconv_examples.mpi_faust.tensorflow.faust_data_set import faust_generator

from tqdm import tqdm

import numpy as np
import os
import shutil


def sc_to_seg_converter(dataset_path, new_dataset_path, segmentation_labels_path):
    """Converts the ground truth labels of the shape-correspondence problem to segmentation labels.

    Parameters
    ----------
    dataset_path: str
        Path to the shape-correspondence dataset.
    new_dataset_path: str
        Path to where the segmentation dataset shall be stored.
    segmentation_labels_path: str
        Path to where the segmentation-labels-array is stored.
    """
    new_dataset_path = os.path.normpath(new_dataset_path)
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    dataset = faust_generator(dataset_path, set_type=3, only_signal=False, return_coordinates=True)
    segmentation_labels = np.load(segmentation_labels_path)

    file_numbers = ["".join(["0" for _ in range(3 - len(f"{i}"))]) + f"{i}" for i in range(100)]
    for idx, ((signal, bc, coord), gt) in tqdm(enumerate(dataset)):
        signal = np.array(signal)
        bc = np.array(bc)
        gt_seg = segmentation_labels[np.array(gt)]

        np.save(f"{new_dataset_path}/SIGNAL_tr_reg_{file_numbers[idx]}.npy", signal)
        np.save(f"{new_dataset_path}/BC_tr_reg_{file_numbers[idx]}.npy", bc)
        np.save(f"{new_dataset_path}/GT_tr_reg_{file_numbers[idx]}.npy", gt_seg)
        np.save(f"{new_dataset_path}/COORD_tr_reg_{file_numbers[idx]}.npy", coord)

    print("Compress converted dataset...")
    shutil.make_archive(new_dataset_path, "zip", new_dataset_path)
    shutil.rmtree(new_dataset_path)
    print("Converting finished.")
