import numpy as np
import scipy as sp
import tensorflow as tf


def faust_generator(path_to_zip, sparse=True, val=False):
    """Reads one element of preprocessed FAUST-datasets into memory per 'next'-call."""

    dataset = np.load(path_to_zip)
    file_names = dataset.files
    SHOT = [file_name for file_name in file_names if file_name.startswith("SHOT")]
    BC = [file_name for file_name in file_names if file_name.startswith("BC")]
    GT = [file_name for file_name in file_names if file_name.startswith("GT")]
    SHOT.sort(), BC.sort(), GT.sort()

    if val:
        indices = range(80, 100)
    else:
        indices = range(80)

    for idx in indices:
        shot = tf.cast(dataset[SHOT[idx]], tf.float32)
        bc = tf.cast(dataset[BC[idx]], tf.float32)

        if sparse:
            # Return the indices of the ones for each row
            # (as required by `tf.keras.losses.SparseCategoricalCrossentropy`)
            gt = sp.sparse.csc_array(dataset[GT[idx]])
            gt = tf.cast(gt.nonzero()[1], tf.float32)
        else:
            gt = dataset[GT[idx]]

        yield (shot, bc), gt


def load_preprocessed_faust(path_to_zip, amt_vertices, kernel_size=(2, 4), sparse=True, val=False):
    """Returns a 'tf.data.Dataset' of the preprocessed MPI-FAUST datasets.

    Requires that preprocessing already happened. This function operates directly on the resulting 'zip'-file.

    **Input**

    - The path to the 'zip'-file obtained through preprocessing.

    **Output**

    - A tf.data.Dataset of the preprocessed MPI-FAUST datasets.

    """

    return tf.data.Dataset.from_generator(
        faust_generator,
        args=(path_to_zip, sparse, val),
        output_signature=(
            (
                tf.TensorSpec(shape=(amt_vertices, 1056), dtype=tf.float32),
                tf.TensorSpec(shape=(amt_vertices,) + kernel_size[::-1] + (6,), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(amt_vertices,), dtype=tf.float32)
        )
    )