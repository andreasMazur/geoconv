from geoconv_examples.mpi_faust.data.preprocess_faust import get_file_number

import numpy as np
import tensorflow as tf
import os
import random


def faust_generator(path_to_zip, set_type=0, only_signal=False, return_coordinates=False, set_indices=None):
    """Reads one element of preprocessed FAUST-geoconv_examples into memory per 'next'-call.

    Parameters
    ----------
    path_to_zip: str
        The path to the .zip-file that contains the preprocessed faust data set
    set_type: int
        This integer has to be either:
            - 0 -> "train"
            - 1 -> "validation"
            - 2 -> "test"
            - 3 -> "all"
        Depending on the choice, the training-, validation or testing data set will be returned. The split is equal to
        the one given in:
        > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://arxiv.org/abs/1501.06297)
        > Jonathan Masci and Davide Boscaini et al.
    only_signal: bool
        Return only the signal matrices. Helpful for keras.Normalization(axis=-1).adapt(data)
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
        signal = tf.cast(dataset[SIGNAL[idx]], tf.float32)

        # Read bc + add noise
        bc = tf.cast(dataset[BC[idx]], tf.float32)
        if kernel_size is None:
            kernel_size = bc.shape[1:3]

        if set_type == 0:
            noise = np.abs(np.random.normal(size=(6890,) + kernel_size + (3, 2), scale=1e-5))
            noise[:, :, :, :, 0] = 0
            bc = bc + noise

        # Ground truth: Return the indices of the ones for each row
        # (as required by `keras.losses.SparseCategoricalCrossentropy`)
        gt = dataset[GT[idx]]

        if only_signal:
            yield signal
        else:
            if return_coordinates:
                coord = tf.cast(dataset[COORD[idx]], tf.float32)
                yield (signal, bc, coord), gt
            else:
                yield (signal, bc), gt


def load_preprocessed_faust(path_to_zip,
                            signal_dim,
                            kernel_size=(2, 4),
                            set_type=0,
                            only_signal=False,
                            return_coordinates=False):
    """Returns a 'tensorflow.data.Dataset' of the preprocessed MPI-FAUST geoconv_examples.

    Requires that preprocessing already happened. This function operates directly on the resulting 'zip'-file.

    Parameters
    ----------
    path_to_zip: str
        The path to the 'zip'-file obtained through preprocessing
    signal_dim: int
        The dimensionality of the defined signal
    kernel_size: tuple
        The same amount of radial- and angular coordinates of the kernel defined in the preprocessing step
    set_type: int
        This integer has to be either:
            - 0 -> "train"
            - 1 -> "validation"
            - 2 -> "test"
        Depending on the choice, the training-, validation or testing data set will be returned. The split is equal to
        the one given in:
        > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://arxiv.org/abs/1501.06297)
        > Jonathan Masci and Davide Boscaini et al.
    only_signal: bool
        Return only the signal matrices. Helpful for keras.Normalization(axis=-1).adapt(data)
    return_coordinates: bool
        Whether to return the coordinates of the mesh vertices. Requires coordinates to be contained in preprocessed
        dataset.

    Returns
    -------
    tensorflow.data.Dataset:
        A tensorflow data set of the preprocessed MPI-FAUST geoconv_examples
    """
    if only_signal:
        output_signature = tf.TensorSpec(shape=(None, signal_dim,), dtype=tf.float32)
    else:
        if return_coordinates:
            output_signature = (
                (
                    tf.TensorSpec(shape=(None, signal_dim,), dtype=tf.float32),  # Signal
                    tf.TensorSpec(shape=(None,) + kernel_size + (3, 2), dtype=tf.float32),  # Barycentric Coordinates
                    tf.TensorSpec(shape=(None, 3,), dtype=tf.float32),  # Coordinates
                ),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        else:
            output_signature = (
                (
                    tf.TensorSpec(shape=(None, signal_dim,), dtype=tf.float32),  # Signal
                    tf.TensorSpec(shape=(None,) + kernel_size + (3, 2), dtype=tf.float32)  # Barycentric Coordinates
                ),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
    return tf.data.Dataset.from_generator(
        faust_generator,
        args=(path_to_zip, set_type, only_signal, return_coordinates),
        output_signature=output_signature
    )
