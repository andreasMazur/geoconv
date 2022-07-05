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
    # label_matrix = np.expand_dims(np.eye(6890), axis=0)
    label_matrix = np.ones((1, 6890, 1))

    for idx in range(100):
        shot = dataset[SHOT[idx]][:, :1]
        shot = np.expand_dims(shot, axis=0)
        # gpc = dataset[GPC[idx]] not required as input in the GCNN
        bc = np.expand_dims(dataset[BC[idx]], axis=0)
        yield shot, bc, label_matrix


def load_preprocessed_faust(path_to_zip):
    """Returns a 'tf.data.Dataset' of the preprocessed MPI-FAUST dataset.

    Requires that preprocessing already happened. This function operates directly on the resulting 'zip'-file.

    **Input**

    - The path to the 'zip'-file obtained through preprocessing.

    **Output**

    - A tf.data.Dataset of the preprocessed MPI-FAUST dataset.

    """

    return tf.data.Dataset.from_generator(
        faust_generator,
        args=(path_to_zip,),
        output_signature=(
            tf.TensorSpec(shape=(1, 6890, 1), dtype=tf.float64),
            # tf.TensorSpec(shape=(6890, 6890, 2), dtype=tf.float64),
            tf.TensorSpec(shape=(1, 6890, 8, 8), dtype=tf.float32),
            tf.TensorSpec(shape=(1, 6890, 1), dtype=tf.int32)
        )
    )


if __name__ == "__main__":
    tf_faust_dataset = load_preprocessed_faust("/home/andreas/Uni/Masterarbeit/MPI-FAUST/preprocessed_faust.zip")
    for elem in tf_faust_dataset:
        print(elem[0].shape, elem[1].shape, elem[2].shape)
        break
