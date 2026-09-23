import tensorflow as tf


def compute_vrmse(y_true, y_pred, axis=(1, 2)):
    """Computes the variance-scaled root mean squared error.

    Parameters
    ----------
    y_true: tf.Tensor | np.ndarray
        The y true tensor.
    y_pred: tf.Tensor | np.ndarray
        The y pred tensor.
    axis: tuple
        The axis.

    Returns
    -------
    tf.Tensor:
        The computed tensor.
    """
    # [n_batch, n_vertices, n_channels]
    squared_difference = tf.math.squared_difference(y_pred, y_true)

    # [n_batch,]
    rmse = tf.math.sqrt(tf.math.reduce_mean(squared_difference, axis=axis))

    # [n_batch,]
    std = tf.math.reduce_std(y_true, axis=axis)

    # [n_batch,]
    return tf.math.divide_no_nan(rmse, std)


class VRMSE(tf.keras.metrics.Metric):
    """Implements the variance-scale root mean squared error."""
    def __init__(self, aggregation_axes=(1, 2), batch_dim=0, name="vrmse", dtype=tf.float32):
        """Initializes the VRMSE metric.
            
        Parameters
        ----------
        aggregation_axes: tuple
            The aggregation axes.
        batch_dim: int
            The batch dimension.
        name: str
            The name.
        dtype: class
            The dtype.
        """
        super(VRMSE, self).__init__(name=name, dtype=dtype)
        self.axis = aggregation_axes
        self.batch_dim = batch_dim
        self.total_vrmse = self.add_weight(name="total_vrmse", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates VRMSE values for the current batch.

        Parameters
        ----------
        y_true: tf.Tensor
            The y true tensor.
        y_pred: tf.Tensor
            The y pred tensor.
        sample_weight: None
            The sample weight. Not used in this class.
        """
        # [n_batch,]
        vrmse = compute_vrmse(y_true, y_pred, axis=self.axis)

        # Update total vrmse
        self.total_vrmse.assign_add(tf.reduce_sum(vrmse))

        # Update counter
        batch_size = tf.cast(tf.shape(y_pred)[self.batch_dim], tf.float32)
        self.count.assign_add(batch_size)

    def result(self):
        """RReturns the mean VRMSE over all observed samples."""
        return self.total_vrmse / self.count
