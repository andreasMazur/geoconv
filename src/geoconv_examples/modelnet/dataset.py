import io
import numpy as np
import h5py
import tensorflow as tf


def get_content(h5_file, kernel_size):
    # Load mesh
    vertices = np.array(h5_file["triangle_mesh/vertices"])

    # Load barycentric coordinates
    barycentric_coordinates = np.array(h5_file["barycentric_coordinates"][f"{kernel_size[0]}_{kernel_size[1]}"])

    # Load parallel transport angles
    parallel_transport = np.array(h5_file["parallel_transport/transport_angles"])

    # Load custom arrays
    gt = np.array(h5_file["custom_arrays"]["ground_truth"])
    return vertices, barycentric_coordinates, parallel_transport, gt


def generator(path, set_type, n_radial, n_angular, return_rotations=True):
    if isinstance(path, bytes):
        set_type = set_type.decode("utf-8")

    zip_file = np.load(path, allow_pickle=True)

    # Load zip content
    zip_content = [f for f in zip_file.files if f.endswith("hdf5")]
    zip_content.sort()

    # Get desired set type
    if set_type == "train":
        zip_content = [f for f in zip_content if f.split("/")[2] == "train"]
    elif set_type == "validation":
        zip_content = [f for f in zip_content if f.split("/")[2] == "validation"]
    elif set_type == "test":
        zip_content = [f for f in zip_content if f.split("/")[2] == "test"]
    elif set_type == "all":
        pass
    else:
        raise RuntimeError(f"Invalid set_type: '{set_type}'. Select either 'train', 'validation', 'test' or 'all'.")

    # Yield dataset elements
    for filepath in zip_content:
        h5_file = h5py.File(io.BytesIO(zip_file[filepath]))
        vertices, barycentric_coordinates, parallel_transport, gt = get_content(h5_file, (n_radial, n_angular))
        if return_rotations:
            yield (vertices, barycentric_coordinates, parallel_transport), gt
        else:
            yield (vertices, barycentric_coordinates), gt


def dataset(path, set_type, n_radial, n_angular, return_rotations=True):
    return tf.data.Dataset.from_generator(
        generator,
        args=(path, set_type, n_radial, n_angular, return_rotations),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,) + (n_radial, n_angular) + (3, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )
    ).prefetch(tf.data.AUTOTUNE)
