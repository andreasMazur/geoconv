from geoconv.utils.data_generator import barycentric_coordinates_generator

import tensorflow as tf
import tensorflow_datasets as tfds


def load_preprocessed_mnist(dataset_path, n_radial, n_angular, template_radius, set_type, batch_size=8):
    """Adds barycentric coordinates to the MNIST dataset and reshapes images to vectors.

    Parameters
    ----------
    dataset_path: str
        The path to the preprocessed dataset.
    n_radial: int
        The amount of radial coordinates used during BC-computation.
    n_angular: int
        The amount of angular coordinates used during BC-computation.
    template_radius: float
        The considered template radius during BC-computation.
    set_type: tensorflow_datasets.SplitArg
        The set type. Either 'train' or 'test'.
    batch_size: int
        The batch-size.

    Returns
    -------
    tensorflow.data.Dataset:
        A dataset containing MNIST-images and labels together with barycentric coordinates to train an IMCNN.
    """
    # Load split MNIST
    splitted_datasets = tfds.load("mnist", split=set_type, shuffle_files=True, as_supervised=True)
    if isinstance(splitted_datasets, list):
        dataset = splitted_datasets[0]
        for d in splitted_datasets[1:]:
            dataset = dataset.concatenate(d)
    else:
        dataset = splitted_datasets

    # Load barycentric coordinates
    barycentric_coordinates = barycentric_coordinates_generator(
        dataset_path, n_radial, n_angular, template_radius, batch_size=1, return_filename=False
    )

    for bc in barycentric_coordinates:
        bc = tf.cast(tf.constant(bc), tf.float32)

        def make_compatible(image, label):
            image = tf.cast(tf.reshape(image, (-1, 1)), tf.float32)
            label = tf.cast(label, tf.int32)
            # Image normalization, adding barycentric coordinates and adjusting data types
            return (tf.cast(image, tf.float32) / 255., bc), label

        # Apply 'make_compatible' to each element of MNIST
        dataset = dataset.map(make_compatible)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
