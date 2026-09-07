import tensorflow as tf


class NormalizePointCloud(tf.keras.layers.Layer):
    """Normalizes a point cloud such that its largest axis has length 1."""
    def call(self, inputs, *args, **kwargs):
        """Applies the layer to the inputs.
            
        Parameters
        ----------
        inputs: tf.Tensor
            A tensor of shape 'b x n x 3' containing the 3D vertex coordinates, whereby 'b' represents the number of
            shapes, 'n' the number of vertices per shape.
        *args: tuple
            The args.
        **kwargs: dict
            The kwargs.

        Returns
        -------
        tf.Tensor
            The normalized point clouds.
        """
        # Move point-cloud into origin
        inputs = inputs - tf.reduce_mean(inputs, axis=1, keepdims=True)

        # Get axis-aligned bounding box
        aabb = tf.reduce_max(inputs, axis=1) - tf.reduce_min(inputs, axis=1)

        # Scale point-cloud so that largest axis has length 1
        return inputs / tf.reduce_max(aabb), aabb
