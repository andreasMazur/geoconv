import sys

import numpy as np
import tensorflow as tf


def faust_mean_variance(faust_dataset, amt_nodes):
    coordinates = []
    x = 0
    for elem in faust_dataset.batch(amt_nodes):
        sys.stdout.write(f"\rLoading triangle mesh {x} / 100 for normalization..")
        coordinates.append(elem[0][0])
        x += 1
    coordinates = tf.stack(coordinates)
    print("Done!")
    return tf.math.reduce_mean(coordinates), tf.math.reduce_variance(coordinates)


def faust_generator(path_to_zip, val=False):
    """Reads one element of preprocessed FAUST-examples into memory per 'next'-call."""

    dataset = np.load(path_to_zip, allow_pickle=True)
    file_names = dataset.files
    SHOT = [file_name for file_name in file_names if file_name.startswith("COORDS")]
    BC = [file_name for file_name in file_names if file_name.startswith("BC")]
    GT = [file_name for file_name in file_names if file_name.startswith("GT")]
    SHOT.sort(), BC.sort(), GT.sort()

    if val:
        indices = range(90, 100)
    else:
        indices = range(90)

    for idx in indices:
        coordinates = tf.cast(dataset[SHOT[idx]], tf.float32)
        bc = tf.cast(dataset[BC[idx]], tf.float32)
        # Return the indices of the ones for each row
        # (as required by `tf.keras.losses.SparseCategoricalCrossentropy`)
        gt = dataset[GT[idx]]
        amt_nodes = tf.shape(coordinates)[0]
        for node_idx in range(amt_nodes):
            yield (coordinates[node_idx], bc[node_idx]), gt[node_idx]


def load_preprocessed_faust(path_to_zip,
                            signal_dim,
                            kernel_size=(2, 4),
                            val=False):
    """Returns a 'tf.data.Dataset' of the preprocessed MPI-FAUST examples.

    Requires that preprocessing already happened. This function operates directly on the resulting 'zip'-file.

    **Input**

    - The path to the 'zip'-file obtained through preprocessing.

    **Output**

    - A tf.data.Dataset of the preprocessed MPI-FAUST examples.

    """

    return tf.data.Dataset.from_generator(
        faust_generator,
        args=(path_to_zip, val),
        output_signature=(
            (
                tf.TensorSpec(shape=(signal_dim,), dtype=tf.float32),
                tf.TensorSpec(shape=kernel_size + (3, 2), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
