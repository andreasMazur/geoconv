from geoconv_examples.mpi_faust.pytorch.faust_data_set import faust_generator

from torch.utils.data import IterableDataset

import numpy as np
import trimesh
import os


PARTNET_LEN = 102
PARTNET_SPLITS = {
    0: list(range(70)),  # train
    1: list(range(70, 80)),  # validation
    2: list(range(80, PARTNET_LEN)),  # test
    3: list(range(PARTNET_LEN))  # all
}


def raw_partnet_grasp_generator(path, return_file_name=False, file_boundaries=None):
    """Loads raw PartNet-Grasp partnet_grasp."""
    directory = os.listdir(f"{path}/out_data")
    directory.sort()
    if file_boundaries is not None:
        directory = directory[file_boundaries[0]:file_boundaries[1]]
    for file_name in directory:
        d = np.load(f'{path}/out_data/{file_name}')
        if return_file_name:
            yield trimesh.Trimesh(vertices=d["verts"], faces=d["faces"], validate=True), d["labels"], file_name
        else:
            yield trimesh.Trimesh(vertices=d["verts"], faces=d["faces"], validate=True), d["labels"]


def processed_partnet_grasp_generator(path_to_zip, set_type=0, only_signal=False, set_indices=None, device=None):
    """Loads preprocessed PartNet-Grasp partnet_grasp."""
    return faust_generator(
        path_to_zip,
        set_type=set_type,
        only_signal=only_signal,
        device=device,
        return_coordinates=False,
        set_indices=PARTNET_SPLITS[set_type] if set_indices is None else set_indices
    )


class PartNetGraspDataset(IterableDataset):
    def __init__(self, path_to_zip, set_type=0, only_signal=False, set_indices=None, device=None):
        self.only_signal = only_signal
        self.path_to_zip = path_to_zip
        self.set_type = set_type
        self.only_signal = only_signal
        self.device = device
        self.set_indices = set_indices

        self.dataset = processed_partnet_grasp_generator(
            self.path_to_zip,
            set_type=self.set_type,
            only_signal=self.only_signal,
            set_indices=self.set_indices,
            device=self.device,
        )

    def __iter__(self):
        return self.dataset

    def reset(self):
        self.dataset = processed_partnet_grasp_generator(
            self.path_to_zip,
            set_type=self.set_type,
            only_signal=self.only_signal,
            set_indices=self.set_indices,
            device=self.device
        )
