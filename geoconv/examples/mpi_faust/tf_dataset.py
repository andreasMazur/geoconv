import numpy as np
import tensorflow as tf


def faust_mean_variance(faust_dataset):
    shot_list = []
    for elem in faust_dataset:
        shot_list.append(elem[0][0])
    shot_list = tf.stack(shot_list)
    return tf.math.reduce_mean(shot_list), tf.math.reduce_variance(shot_list)


def faust_generator(path_to_zip, val=False):
    """Reads one element of preprocessed FAUST-examples into memory per 'next'-call."""

    dataset = np.load(path_to_zip, allow_pickle=True)
    file_names = dataset.files
    SHOT = [file_name for file_name in file_names if file_name.startswith("SHOT")]
    BC = [file_name for file_name in file_names if file_name.startswith("BC")]
    GT = [file_name for file_name in file_names if file_name.startswith("GT")]
    SHOT.sort(), BC.sort(), GT.sort()

    if val:
        indices = range(90, 100)
    else:
        indices = range(90)

    for idx in indices:
        shot = tf.cast(dataset[SHOT[idx]], tf.float32)
        bc = tf.cast(dataset[BC[idx]], tf.float32)
        gt = dataset[GT[idx]][()]
        # Return the indices of the ones for each row
        # (as required by `tf.keras.losses.SparseCategoricalCrossentropy`)
        gt = tf.cast(gt.nonzero()[1], tf.float32)

        yield (shot, bc), gt


def load_preprocessed_faust(path_to_zip,
                            amt_vertices,
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
                tf.TensorSpec(shape=(amt_vertices, signal_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(amt_vertices,) + kernel_size[::-1] + (6,), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(amt_vertices,), dtype=tf.float32)
        )
    )
