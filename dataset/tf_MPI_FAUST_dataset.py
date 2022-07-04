import numpy as np
import tensorflow as tf


def faust_generator(path_to_zip):
    """Reads one element of preprocessed FAUST-dataset into memory per 'next'-call."""

    dataset = np.load(path_to_zip)
    file_names = dataset.files
    SHOT = [file_name for file_name in file_names if file_name.startswith("SHOT")]
    GPC = [file_name for file_name in file_names if file_name.startswith("GPC")]
    BC = [file_name for file_name in file_names if file_name.startswith("BC")]
    SHOT.sort(), GPC.sort(), BC.sort()
    for idx in range(100):
        shot = dataset[SHOT[idx]]
        # gpc = dataset[GPC[idx]] not required as input in the GCNN
        bc = dataset[BC[idx]]
        yield shot, bc


def load_preprocessed_faust(path_to_zip):
    """Returns a 'tf.data.Dataset' of the preprocessed MPI-FAUST dataset.

    Requires that preprocessing already happened. This function operates directly on the resulting 'zip'-file.

    **Input**

    - The path to the 'zip'-file obtained through preprocessing.

    **Output**

    - A tf.data.Dataset of the preprocessed MPI-FAUST dataset.

    """

    gen = faust_generator(path_to_zip)
    return tf.data.Dataset.from_generator(
        lambda: gen,  # "gen must be callable"
        output_signature=(
            tf.TensorSpec(shape=(6890, 1056), dtype=tf.float64),
            # tf.TensorSpec(shape=(6890, 6890, 2), dtype=tf.float64),
            tf.TensorSpec(shape=(6890, 8, 8), dtype=tf.float32)
        )
    )


if __name__ == "__main__":
    tf_faust_dataset = load_preprocessed_faust("/home/andreas/Uni/Masterarbeit/MPI-FAUST/preprocessed_faust.zip")
