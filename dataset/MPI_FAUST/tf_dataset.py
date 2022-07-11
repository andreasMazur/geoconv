import numpy as np
import tensorflow as tf
import trimesh
import zipfile


def faust_generator(path_to_zip):
    """Reads one element of preprocessed FAUST-dataset into memory per 'next'-call."""

    dataset = np.load(path_to_zip)
    file_names = dataset.files
    SHOT = [file_name for file_name in file_names if file_name.startswith("SHOT")]
    BC = [file_name for file_name in file_names if file_name.startswith("BC")]
    GT = [file_name for file_name in file_names if file_name.startswith("GT")]
    SHOT.sort(), BC.sort(), GT.sort()

    # z = zipfile.ZipFile(path_to_zip)
    # PLY = [file_name for file_name in file_names if file_name.endswith("ply")]
    # GPC = [file_name for file_name in file_names if file_name.startswith("GPC")]
    # GPC.sort(), PLY.sort()

    for idx in range(100):
        shot = np.expand_dims(dataset[SHOT[idx]], axis=0)
        bc = np.expand_dims(dataset[BC[idx]], axis=0)
        gt = np.expand_dims(dataset[GT[idx]], axis=0)

        # not required as input in the GCNN
        # gpc = dataset[GPC[idx]]
        # ply = trimesh.exchange.ply.load_ply(z.open(PLY[idx]))
        # ply = trimesh.Trimesh(vertices=ply["vertices"], faces=ply["faces"])

        yield (shot, bc), gt
        # yield (shot, bc, gpc, ply), gt


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
            (
                tf.TensorSpec(shape=(1, 6890, 1056), dtype=tf.float32),
                tf.TensorSpec(shape=(1, 6890, 8, 8), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(1, 6890, 6890), dtype=tf.int32)
        )
    )


if __name__ == "__main__":
    tf_faust_dataset = load_preprocessed_faust(
        "/home/andreas/PycharmProjects/Masterarbeit/dataset/MPI_FAUST/preprocessed_registrations.zip"
    )
    # faust_dataset = faust_generator(
    #     "/home/andreas/PycharmProjects/Masterarbeit/dataset/MPI_FAUST/preprocessed_registrations.zip"
    # )
    for elem in tf_faust_dataset:
        print("Signal:", elem[0][0].shape, elem[0][0].dtype)
        print("Barycentric coordinates:", elem[0][1].shape, elem[0][1].dtype)
        print("Labels:", elem[1].shape, elem[1].dtype)
        break
