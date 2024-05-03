import numpy as np
import trimesh
import os


def raw_data_generator(path, return_file_name=False, file_boundaries=None):
    """Loads the manually labeled data."""
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


def processed_data_generator(path_to_zip, set_type=0, only_signal=False, device=None, return_coordinates=False):
    """TODO: Reads one element of preprocessed FAUST-geoconv_examples into memory per 'next'-call.

    Parameters
    ----------
    path_to_zip: str
        The path to the .zip-file that contains the preprocessed faust data set
    set_type: int
        This integer has to be either:
            - 0 -> "train"
            - 1 -> "validation"
            - 2 -> "test"
            - 3 -> "all meshes"
        Depending on the choice, the training-, validation or testing data set will be returned. The split is equal to
        the one given in:
        > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://arxiv.org/abs/1501.06297)
        > Jonathan Masci and Davide Boscaini et al.
    only_signal: bool
        Return only the signal matrices. Helpful for tensorflow.keras.Normalization(axis=-1).adapt(data)
    device:
        The device to put the data on.
    return_coordinates: bool
        Whether to return the coordinates of the mesh vertices. Requires coordinates to be contained in preprocessed
        dataset.

    Returns
    -------
    generator:
        A generator yielding the preprocessed data. I.e. the signal defined on the vertices, the barycentric coordinates
        and the ground truth correspondences.
    """
    kernel_size = None
    dataset = np.load(path_to_zip, allow_pickle=True)
    file_names = [os.path.basename(fn) for fn in dataset.files]

    SIGNAL = [file_name for file_name in file_names if file_name.startswith("SIGNAL")]
    BC = [file_name for file_name in file_names if file_name.startswith("BC")]
    GT = [file_name for file_name in file_names if file_name.startswith("GT")]
    SIGNAL.sort(key=get_file_number), BC.sort(key=get_file_number), GT.sort(key=get_file_number)
    if return_coordinates:
        COORD = [file_name for file_name in file_names if file_name.startswith("COORD")]
        COORD.sort(key=get_file_number)

    if set_type == 0:
        indices = list(range(40))
        random.shuffle(indices)
    elif set_type == 1:
        indices = range(40, 50)
    elif set_type == 2:
        indices = range(50, 60)
    elif set_type == 3:
        indices = range(60)
    else:
        raise RuntimeError(f"There is no 'set_type'={set_type}. Choose from: [0: 'train', 1: 'val', 2: 'test'].")

    for idx in indices:
        # Read signal
        signal = torch.tensor(dataset[SIGNAL[idx]], dtype=torch.float32)

        # Read bc + add noise
        bc = torch.tensor(dataset[BC[idx]], dtype=torch.float32)
        if kernel_size is None:
            kernel_size = bc.shape[1:3]

        if set_type == 0 or set_type == 3:
            noise = np.abs(np.random.normal(size=(bc.shape[0],) + kernel_size + (3, 2), scale=1e-5))
            noise[:, :, :, :, 0] = 0
            bc = bc + noise

        # Ground truth: Return the indices of the ones for each row
        gt = torch.tensor(dataset[GT[idx]], dtype=torch.int64)

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
                    yield (signal, bc, SIGNAL[idx].split(".")[0].split("_")[1]), gt
