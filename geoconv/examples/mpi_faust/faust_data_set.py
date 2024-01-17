import numpy as np
import tensorflow as tf
import os


def get_file_number(file_name):
    """Extracts the file number contained in the file name

    Parameters
    ----------
    file_name: str
        The file name

    Returns
    -------
    int:
        The file number contained in the file name
    """
    # file_name.split(".")[0] -> Without file ending
    for elem in file_name.split(".")[0].split("_"):
        if elem.isdigit():
            return int(elem)
    raise RuntimeError(f"Filename '{file_name}' has no digit.")


def faust_generator(path_to_zip, set_type=0, only_signal=False):
    """Reads one element of preprocessed FAUST-examples into memory per 'next'-call.

    Parameters
    ----------
    path_to_zip: str
        The path to the .zip-file that contains the preprocessed faust data set
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
        Return only the signal matrices. Helpful for tf.keras.Normalization(axis=-1).adapt(data)

    Returns
    -------
    generator:
        A generator yielding the preprocessed data. I.e. the signal defined on the vertices, the barycentric coordinates
        and the ground truth correspondences.
    """
    dataset = np.load(path_to_zip, allow_pickle=True)
    file_names = [os.path.basename(fn) for fn in dataset.files]
    SIGNAL = [file_name for file_name in file_names if file_name.startswith("SIGNAL")]
    BC = [file_name for file_name in file_names if file_name.startswith("BC")]
    GT = [file_name for file_name in file_names if file_name.startswith("GT")]
    SIGNAL.sort(key=get_file_number), BC.sort(key=get_file_number), GT.sort(key=get_file_number)

    if set_type == 0:
        indices = range(70)
    elif set_type == 1:
        indices = range(70, 80)
    elif set_type == 2:
        indices = range(80, 100)
    else:
        raise RuntimeError(f"There is no 'set_type'={set_type}. Choose from: [0: 'train', 1: 'val', 2: 'test'].")

    for idx in indices:
        signal = tf.cast(dataset[SIGNAL[idx]], tf.float32)
        bc = tf.cast(dataset[BC[idx]], tf.float32)
        # Return the indices of the ones for each row
        # (as required by `tf.keras.losses.SparseCategoricalCrossentropy`)
        gt = dataset[GT[idx]]
        if only_signal:
            yield signal
        else:
            yield (signal, bc), gt


def load_preprocessed_faust(path_to_zip,
                            signal_dim,
                            kernel_size=(2, 4),
                            set_type=0,
                            only_signal=False):
    """Returns a 'tf.data.Dataset' of the preprocessed MPI-FAUST examples.

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
        Return only the signal matrices. Helpful for tf.keras.Normalization(axis=-1).adapt(data)

    Returns
    -------
    tf.data.Dataset:
        A tensorflow data set of the preprocessed MPI-FAUST examples
    """
    if only_signal:
        output_signature = tf.TensorSpec(shape=(None, signal_dim,), dtype=tf.float32)
    else:
        output_signature = (
            (
                tf.TensorSpec(shape=(None, signal_dim,), dtype=tf.float32),  # Signal
                tf.TensorSpec(shape=(None,) + kernel_size + (3, 2), dtype=tf.float32),  # Barycentric Coordinates
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    return tf.data.Dataset.from_generator(
        faust_generator,
        args=(path_to_zip, set_type, only_signal),
        output_signature=output_signature
    )
