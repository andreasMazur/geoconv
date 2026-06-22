from geoconv.preprocessing.atlas import load_atlas

import tensorflow as tf
import tensorflow_datasets as tfds


def dataset(mnist_atlas, set_type, n_radial, n_angular, radius, batch_size, return_rotations=True):
    """The MNIST dataset adjusted for use in combination with surface CNNs.

    Parameters
    ----------
    mnist_atlas: Atlas
        An Atlas for the 28x28 MNIST picture.
    set_type: str
        Either "train" or "test".
    n_radial: int
        The amount of radial coordinates used by the discretized template.
    n_angular: int
        The amount of angular coordinates used by the discretized template.
    radius: float
        The radius of the discretized template.
    batch_size: int
        The batch size.
    return_rotations: bool
        Whether to return rotations for a parallel transport.

    Returns
    -------
    tf.dataset.Dataset:
        An MNIST dataset.
    """
    # Load barycentric coordinates
    atlas = load_atlas(mnist_atlas)
    bc = atlas.barycentric_coordinates[(n_radial, n_angular, radius)]

    # Load images
    mnist = tfds.load("mnist", split=set_type, shuffle_files=True, as_supervised=True)

    # Set default rotations, rotation order and imaginary values for every image
    rotations = tf.zeros((784, 784))
    imaginary_values = tf.zeros((784, 1), dtype=tf.float32)

    # Return image, barycentric coordinates and label
    def transform(image, label):
        # Normalize image
        image = tf.cast(image, tf.float32) / 255.

        # Lift 1-d features into imaginary plane by mapping 'grey_scale_value -> grey_scale_value + 0i'
        image = tf.reshape(image, (784, 1))
        image = tf.concat([image, imaginary_values], axis=-1)
        if return_rotations:
            return (image, bc, rotations), label
        else:
            return (image, bc), label
    mnist = mnist.map(transform)

    # Return batched MNIST
    return mnist.batch(batch_size).prefetch(tf.data.AUTOTUNE)
