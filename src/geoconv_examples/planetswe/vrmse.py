import tensorflow as tf


def compute_vrmse(y_true, y_pred, axis=(1, 2)):
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
    def __init__(self, aggregation_axes=(1, 2), batch_dim=0, name="vrmse"):
        super(VRMSE, self).__init__(name=name)
        self.axis = aggregation_axes
        self.batch_dim = batch_dim
        self.total_vrmse = self.add_weight(name="total_vrmse", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # [n_batch,]
        vrmse = compute_vrmse(y_true, y_pred, axis=self.axis)

        # Update total vrmse
        self.total_vrmse.assign_add(tf.reduce_sum(vrmse))

        # Update counter
        batch_size = tf.cast(tf.shape(y_pred)[self.batch_dim], tf.float32)
        self.count.assign_add(batch_size)

    def result(self):
        return self.total_vrmse / self.count
