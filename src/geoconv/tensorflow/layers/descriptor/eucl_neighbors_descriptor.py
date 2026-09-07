from geoconv.tensorflow.utils.compute_shot_lrf import compute_neighborhood

import tensorflow as tf


class EuclNeighborsDescriptor(tf.keras.layers.Layer):
    """The Euclidean neighborhood descriptor layer.

    Attributes
    ----------
    n_neighbors : int
        The amount of neighbors to consider for computing local neighborhood descriptors.
    """
    def __init__(self, n_neighbors, *args, **kwargs):
        """Initializes the object.

        Parameters
        ----------
        n_neighbors: int
            The number of neighbors to consider.
        *args: tuple
            The args.
        **kwargs: dict
            The kwargs.
        """
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors

    def call(self, inputs, **kwargs):
        """Computes local Euclidean neighborhood descriptor.

        Parameters
        ----------
        inputs: tf.Tensor
            The vertices of the input shape.

        Returns
        -------
        tf:Tensor:
            A tensor of shape [batch, vertices, 3 * n_neighbors - 3]
        """
        # 'compute_neighborhood' returns neighbors in order to their Euclidean distance to the origin
        # 'neighborhoods' : (batch, vertices, n_neighbors, 3)
        neighborhoods, _, _ = compute_neighborhood(inputs, self.n_neighbors)

        # Scale max sphere radius to 1
        # 'neighborhoods' : (batch, vertices, n_neighbors, 3)
        neighborhoods = tf.math.divide_no_nan(
            neighborhoods, tf.reduce_max(tf.linalg.norm(neighborhoods, axis=-1), axis=-1)[..., None, None]
        )

        neighborhoods_shape = tf.shape(neighborhoods)

        # 'return': (batch, vertices, n_neighbors, 3 * neighbors - 3)
        return tf.reshape(
            neighborhoods, (neighborhoods_shape[0], neighborhoods_shape[1], self.n_neighbors * 3)
        )[..., 3:]  # Cut away the origin-zero vectors
