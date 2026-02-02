import tensorflow as tf
import numpy as np


def generator(zip_path, set_type, n_radial, n_angular, radius, return_rotations=True):
    """Returns a 'generator'-object for the ModelNet dataset.

    Parameters
    ----------
    zip_path: str
        The path to the preprocessed zip-file of ModelNet.
    set_type: str
        The set type. Either: 'train', 'validation', 'test' or 'all'.
    n_radial: int
        The number of radial coordinates of the template.
    n_angular: int
        The number of angular coordinates of the template.
    radius: float
        The radius of the template.
    return_rotations: bool
        Whether to return the rotation angles for the parallel transport.

    Returns
    -------
    generator:
        A ModelNet generator.
    """
    if isinstance(zip_path, bytes):
        set_type = set_type.decode("utf-8")

    zip_file = np.load(zip_path, allow_pickle=True)

    # Load zip content
    zip_content = [f"tr_reg_{i:03d}" for i in range(100)]

    # Get desired set type
    if set_type == "train":
        zip_content = zip_content[:70]
    elif set_type == "validation":
        zip_content = zip_content[70:80]
    elif set_type == "test":
        zip_content = zip_content[80:99]
    elif set_type == "all":
        pass
    else:
        raise RuntimeError(f"Invalid set_type: '{set_type}'. Select either 'train', 'validation', 'test' or 'all'.")

    # Yield dataset elements
    for filepath in zip_content:
        vertices = zip_file[f"{filepath}/vertices"]
        barycentric_coordinates = zip_file[
            f"{filepath}/barycentric_coordinates_{n_radial}_{n_angular}_{'_'.join(f'{radius}'.split('.'))}"
        ]
        ground_truth = zip_file[f"{filepath}/ground_truth"]
        if return_rotations:
            parallel_transport = zip_file[f"{filepath}/parallel_transport"]
            yield (vertices, barycentric_coordinates, parallel_transport), ground_truth
        else:
            yield (vertices, barycentric_coordinates), ground_truth


def dataset(zip_path, set_type, n_radial, n_angular, radius, return_rotations=True):
    """Returns a 'tensorflow dataset'-object for the ModelNet dataset.

    Parameters
    ----------
    zip_path: str
        The path to the preprocessed zip-file of ModelNet.
    set_type: str
        The set type. Either: 'train', 'validation', 'test' or 'all'.
    n_radial: int
        The number of radial coordinates of the template.
    n_angular: int
        The number of angular coordinates of the template.
    radius: tf.Tensor
        The radius of the template.
    return_rotations: bool
        Whether to return the rotation angles for the parallel transport.

    Returns
    -------
    tf.data.Dataset:
        A ModelNet dataset.
    """
    if return_rotations:
        output_signature = (
            (
                tf.TensorSpec(shape=(6890, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(6890,) + (n_radial, n_angular) + (3, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(6890,), dtype=tf.float32),
        )
    else:
        output_signature = (
            (
                tf.TensorSpec(shape=(6890, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(6890,) + (n_radial, n_angular) + (3, 2), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(6890,), dtype=tf.float32),
        )

    return tf.data.Dataset.from_generator(
        generator,
        args=(zip_path, set_type, n_radial, n_angular, radius, return_rotations),
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)
