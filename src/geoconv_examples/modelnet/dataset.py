import numpy as np
import tensorflow as tf
import os


def generator(path,
              set_type,
              chart_max_radius,
              method,
              return_rotations=True):
    """Returns a 'generator'-object for the ModelNet dataset.

    Parameters
    ----------
    path: str
        The path to the preprocessed zip-file of ModelNet.
    set_type: str
        The set type. Either: 'train', 'test' or 'all'.
    chart_max_radius: float
        The upper bound radius for the local charts.
    method: str
        The used preprocessing method.
    return_rotations: bool
        Whether to return the rotation angles for the parallel transport.

    Returns
    -------
    generator:
        A ModelNet generator.
    """
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    if isinstance(set_type, bytes):
        set_type = set_type.decode("utf-8")
    if isinstance(method, bytes):
        method = method.decode("utf-8")

    zip_file = np.load(path, allow_pickle=True)

    # Load zip content
    zip_content = [
        f for f in zip_file.files
        if f"mn10_{method}_{'_'.join(f'{chart_max_radius}'.split('.'))}" in f and "barycentric_coordinates" in f
    ]
    print(f"mn10_{method}_{'_'.join(f'{chart_max_radius}'.split('.'))}")

    # Get desired set type
    if set_type == "train":
        zip_content = [f for f in zip_content if "train" in f]
    elif set_type == "test":
        zip_content = [f for f in zip_content if "test" in f]
    elif set_type == "all":
        pass
    else:
        raise RuntimeError(f"Invalid set_type: '{set_type}'. Select either 'train', 'validation', 'test' or 'all'.")

    for filepath in zip_content:
        # Load barycentric coordinates
        barycentric_coordinates = zip_file[filepath]

        # Load related shape info
        file_dir = os.path.dirname(filepath)
        vertices = zip_file[f"{file_dir}/vertices"]
        gt = zip_file[f"{file_dir}/ground_truth"]

        # Yield dataset elements
        if return_rotations:
            yield (vertices, barycentric_coordinates), gt
        else:
            yield (vertices, barycentric_coordinates[..., :2]), gt


def dataset(path, set_type, n_radial, n_angular, chart_max_radius, method, return_rotations=True):
    """Returns a 'tensorflow dataset'-object for the ModelNet dataset.

    Parameters
    ----------
    path: str
        The path to the preprocessed zip-file of ModelNet.
    set_type: str
        The set type. Either: 'train', 'test' or 'all'.
    n_radial: int
        The number of radial coordinates of the template.
    n_angular: int
        The number of angular coordinates of the template.
    chart_max_radius: tf.Tensor
        The upper bound radius for the local charts.
    method: str
        The used preprocessing method.
    return_rotations: bool
        Whether to return the rotation angles for the parallel transport.

    Returns
    -------
    tf.data.Dataset:
        A ModelNet dataset.
    """
    if return_rotations:
        bc_shape = (3, 3)
    else:
        bc_shape = (3, 2)
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,) + (n_radial, n_angular) + bc_shape, dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )

    return tf.data.Dataset.from_generator(
        generator,
        args=(path, set_type, chart_max_radius, method, return_rotations),
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE).batch(1)
