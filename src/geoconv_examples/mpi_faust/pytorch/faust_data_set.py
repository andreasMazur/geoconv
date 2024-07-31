from geoconv_examples.mpi_faust.data.preprocess_faust import get_file_number

from torch.utils.data import IterableDataset

import torch
import random
import numpy as np
import os


def faust_generator(path_to_zip,
                    set_type=0,
                    only_signal=False,
                    device=None,
                    return_coordinates=False,
                    set_indices=None):
    """Reads one element of preprocessed FAUST-geoconv_examples into memory per 'next'-call.

    Parameters
    ----------
    path_to_zip: str
        The path to the .zip-file that contains the preprocessed faust data set
    set_type: int
        This integer has to be either:
            - 0 -> "train"  (adds noise to barycentric coordinates)
            - 1 -> "validation"
            - 2 -> "test"
            - 3 -> "all meshes"
        Depending on the choice, the training-, validation or testing data set will be returned. The split is equal to
        the one given in:
        > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://arxiv.org/abs/1501.06297)
        > Jonathan Masci and Davide Boscaini et al.
    only_signal: bool
        Return only the signal matrices.
    device:
        The device to put the data on.
    return_coordinates: bool
        Whether to return the coordinates of the mesh vertices. Requires coordinates to be contained in preprocessed
        dataset.
    set_indices: list
        A list of integer values that determine which meshes shall be returned. If it is set to 'None', the set
        type determine which meshes will be returned. Defaults to 'None'. Adds noise to barycentric coordinates
        if set type is set to 0.

    Returns
    -------
    generator:
        A generator yielding the preprocessed data. I.e. the signal defined on the vertices, the barycentric coordinates
        and the ground truth correspondences.
    """
    # Initialize and sort file names
    dataset = np.load(path_to_zip, allow_pickle=True)
    file_names = [os.path.basename(fn) for fn in dataset.files]
    SIGNAL = [file_name for file_name in file_names if file_name.startswith("SIGNAL")]
    BC = [file_name for file_name in file_names if file_name.startswith("BC")]
    GT = [file_name for file_name in file_names if file_name.startswith("GT")]
    SIGNAL.sort(key=get_file_number), BC.sort(key=get_file_number), GT.sort(key=get_file_number)
    if return_coordinates:
        COORD = [file_name for file_name in file_names if file_name.startswith("COORD")]
        COORD.sort(key=get_file_number)

    # Set iteration indices according to set type
    if set_indices is None:
        if set_type == 0:
            indices = list(range(70))
            random.shuffle(indices)
        elif set_type == 1:
            indices = range(70, 80)
        elif set_type == 2:
            indices = range(80, 100)
        elif set_type == 3:
            indices = range(100)
        else:
            raise RuntimeError(
                f"There is no 'set_type'={set_type}. Choose from: [0: 'train', 1: 'val', 2: 'test', 3: 'all']."
            )
    else:
        indices = set_indices

    for idx in indices:
        # Read signal
        signal = torch.tensor(dataset[SIGNAL[idx]], dtype=torch.float32)

        # Read bc + add noise
        bc = torch.tensor(dataset[BC[idx]], dtype=torch.float32)
        kernel_size = bc.shape[1:3]

        if set_type == 0:
            noise = np.abs(np.random.normal(size=(bc.shape[0],) + kernel_size + (3, 2), scale=1e-5))
            noise[:, :, :, :, 0] = 0
            bc = bc + noise

        # Ground truth: Return the indices of the ones for each row
        gt = torch.tensor(dataset[GT[idx]], dtype=torch.int64).view(-1,)

        if device:
            if only_signal:
                yield signal.to(device)
            else:
                yield (signal.to(device), bc.to(device)), gt.to(device)
        else:
            if only_signal:
                yield signal
            else:
                # Coordinates are not required during training. However, other applications might need them.
                if return_coordinates:
                    coord = torch.tensor(dataset[COORD[idx]], dtype=torch.float32)
                    yield (signal, bc, coord), gt
                else:
                    yield (signal, bc), gt


class FaustDataset(IterableDataset):
    def __init__(self, path_to_zip, set_type=0, only_signal=False, device=None, return_coordinates=False):
        self.only_signal = only_signal
        self.path_to_zip = path_to_zip
        self.set_type = set_type
        self.only_signal = only_signal
        self.return_coordinates = return_coordinates
        self.device = device
        self.dataset = faust_generator(
            self.path_to_zip,
            set_type=self.set_type,
            only_signal=self.only_signal,
            device=self.device,
            return_coordinates=self.return_coordinates
        )

    def __iter__(self):
        return self.dataset

    def reset(self):
        self.dataset = faust_generator(
            self.path_to_zip,
            set_type=self.set_type,
            only_signal=self.only_signal,
            device=self.device,
            return_coordinates=self.return_coordinates
        )