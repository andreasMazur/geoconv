from geoconv_examples.mnist.preprocess import create_grid

import tensorflow as tf
import tensorflow_datasets as tfds


def load_preprocessed_mnist_for_projections(set_type, batch_size=8, for_adaptation=False):
    """Loads MNIST while adding 3D coordinates for the image pixels to the dataset and reshapes color values to vectors.

    Parameters
    ----------
    set_type: tensorflow_datasets.SplitArg
        The set type. Either 'all', 'train' or 'test'. Alternatively, a list of 'tfds.typing.SplitArg'.
    batch_size: int
        The batch-size.
    for_adaptation: bool
        If True, the dataset is prepared for BC-layer adaptation, meaning that only the 3D coordinates are returned.

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

    # Create 28x28 grid whose vertices are to be processed by BC-layer to barycentric coordinates
    grid = create_grid(n_vertices=28).vertices

    # Define a function to make the dataset compatible with the IMCNN input requirements.
    # If 'for_adaptation' is True, we only return the grid and the label.
    if for_adaptation:
        return tf.data.Dataset.from_tensors((tf.reshape(tf.cast(grid, tf.float32), (1, -1, 3)), None))
    else:
        def make_compatible(image, label):
            image = tf.cast(tf.reshape(image, (-1, 1)), tf.float32)
            label = tf.cast(label, tf.int32)
            # Image normalization, adding barycentric coordinates and adjusting data types
            return (tf.cast(image, tf.float32) / 255., tf.cast(grid, tf.float32)), label

    # Apply 'make_compatible' to each element of MNIST
    dataset = dataset.map(make_compatible)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
