from geoconv.tensorflow.layers import EuclNeighborsDescriptor

from tqdm import tqdm

import tensorflow as tf
import numpy as np


def adapt_generator(zip_path, set_type, n_radial, n_angular, preprocess_method, gpc_radius, template_radius):
    descr_layer = EuclNeighborsDescriptor(n_neighbors=int(1 + n_radial * n_angular))
    gen = generator(
        zip_path, set_type, n_radial, n_angular, preprocess_method, gpc_radius, template_radius, return_rotations=False
    )
    for (vertices, bc), _ in tqdm(gen, postfix="Adapting normalization layer..."):
        yield descr_layer(vertices[None, ...])[0]


def adapt_dataset(zip_path,
                  set_type,
                  n_radial,
                  n_angular,
                  preprocess_method,
                  gpc_radius,
                  template_radius,
                  layer_output_dim):
    return tf.data.Dataset.from_generator(
        adapt_generator,
        args=(zip_path, set_type, n_radial, n_angular, preprocess_method, gpc_radius, template_radius),
        output_signature=(tf.TensorSpec(shape=(6890, layer_output_dim), dtype=tf.float32))
    ).prefetch(tf.data.AUTOTUNE).batch(1)


def generator(zip_path,
              set_type,
              n_radial,
              n_angular,
              preprocess_method,
              gpc_radius,
              template_radius,
              return_rotations=True):
    """Returns a 'generator'-object for the FAUST dataset.

    Parameters
    ----------
    zip_path: str
        The path to the preprocessed zip-file of FAUST.
    set_type: str
        The set type. Either: 'train', 'validation', 'test' or 'all'.
    n_radial: int
        The number of radial coordinates of the template.
    n_angular: int
        The number of angular coordinates of the template.
    preprocess_method: str
        The used preprocessing method. Either 'fmm', 'dgpc' or 'hdm'.
    gpc_radius: float
        The radius of the GPC system.
    template_radius: float | tf.Tensor
        The radius of the template.
    return_rotations: bool
        Whether to return the rotation angles for the parallel transport.

    Returns
    -------
    generator:
        A FAUST generator.
    """
    if isinstance(zip_path, bytes):
        zip_path = zip_path.decode("utf-8")
    if isinstance(set_type, bytes):
        set_type = set_type.decode("utf-8")
    if isinstance(preprocess_method, bytes):
        preprocess_method = preprocess_method.decode("utf-8")

    zip_file = np.load(zip_path, allow_pickle=True)

    # Load zip content
    zip_content = [
        f"faust_{preprocess_method}_{'_'.join(f'{gpc_radius}'.split('.'))}/tr_reg_{i:03d}" for i in range(100)
    ]

    # Get desired set type
    if set_type == "train":
        zip_content = zip_content[:70]
    elif set_type == "validation":
        zip_content = zip_content[70:80]
    elif set_type == "test":
        zip_content = zip_content[80:100]
    elif set_type == "all":
        pass
    else:
        raise RuntimeError(f"Invalid set_type: '{set_type}'. Select either 'train', 'validation', 'test' or 'all'.")

    # Yield dataset elements
    for filepath in zip_content:
        vertices = zip_file[f"{filepath}/vertices"]
        barycentric_coordinates = zip_file[
            f"{filepath}/barycentric_coordinates_{n_radial}_{n_angular}_{f'{template_radius}'.replace('.', '_')}"
        ]
        ground_truth = zip_file[f"{filepath}/ground_truth"]
        if return_rotations:
            yield (vertices, barycentric_coordinates), ground_truth
        else:
            yield (vertices, barycentric_coordinates[..., :2]), ground_truth


def dataset(zip_path,
            set_type,
            n_radial,
            n_angular,
            preprocess_method,
            gpc_radius,
            template_radius,
            return_rotations=True):
    """Returns a 'tensorflow dataset'-object for the FAUST dataset.

    Parameters
    ----------
    zip_path: str
        The path to the preprocessed zip-file of FAUST.
    set_type: str
        The set type. Either: 'train', 'validation', 'test' or 'all'.
    n_radial: int
        The number of radial coordinates of the template.
    n_angular: int
        The number of angular coordinates of the template.
    preprocess_method: str
        The used preprocessing method. Either 'fmm', 'dgpc' or 'hdm'.
    gpc_radius: tf.Tensor
        The radius of the GPC system.
    template_radius: tf.Tensor
        The radius of the template.
    return_rotations: bool
        Whether to return the rotation angles for the parallel transport.

    Returns
    -------
    tf.data.Dataset:
        A FAUST dataset.
    """
    if return_rotations:
        bc_shape = (3, 3)
    else:
        bc_shape = (3, 2)
    output_signature = (
        (
            tf.TensorSpec(shape=(6890, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(6890,) + (n_radial, n_angular) + bc_shape, dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(6890,), dtype=tf.float32),
    )

    return tf.data.Dataset.from_generator(
        generator,
        args=(
            zip_path, set_type, n_radial, n_angular, preprocess_method, gpc_radius, template_radius, return_rotations
        ),
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE).batch(1)
