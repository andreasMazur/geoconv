import tensorflow as tf


class VRMSE(tf.keras.metrics.Metric):
    """Implements the variance-scale root mean squared error."""
    def __init__(self, axis=(1, 2), name="vrmse"):
        super(VRMSE, self).__init__(name=name)
        self.axis = axis
        self.total_vrmse = self.add_weight(name="total_vrmse", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # [n_batch, n_vertices, n_channels]
        squared_difference = tf.math.squared_difference(y_pred, y_true)

        # [n_batch,]
        rmse = tf.math.sqrt(tf.math.reduce_mean(squared_difference, axis=self.axis))

        # [n_batch,]
        std = tf.math.reduce_std(y_true, axis=self.axis, keepdims=True)

        # [n_batch,]
        return rmse / std

    def result(self):
        return self.total_vrmse / self.count
