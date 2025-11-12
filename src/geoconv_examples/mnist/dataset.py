from geoconv.preprocessing.atlas import load_atlas

import tensorflow as tf
import tensorflow_datasets as tfds


def dataset(mnist_atlas, set_type, n_radial, n_angular, batch_size):
    # Load barycentric coordinates
    atlas = load_atlas(mnist_atlas)
    bc = atlas.barycentric_coordinates[(n_radial, n_angular)]

    # Load images
    mnist = tfds.load("mnist", split=set_type, shuffle_files=True, as_supervised=True)

    # Return image, barycentric coordinates and label
    def transform(image, label):
        return (tf.reshape(tf.cast(image, tf.float32), (-1, 1)), bc), label
    mnist = mnist.map(transform)

    # Return batched MNIST
    return mnist.batch(batch_size).prefetch(tf.data.AUTOTUNE)
