from tensorflow import keras

import tensorflow as tf


class ImCNN(keras.Model):

    def __init__(self, *args, max_rotations, splits=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.splits = splits if splits is not None else 1
        self.concurrent_rotations = tf.split(tf.range(max_rotations), self.splits)

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # Update evaluation metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            elif metric.name in self.gradient_stat_names:
                pass
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics if m.name not in self.gradient_stat_names}

    def train_step(self, data):
        x, y = data
        y_pred, loss = self.gradient_step(x, y)

        # Initialize metrics with first observation
        if not self.compiled_metrics.built:
            self.compiled_metrics.build(y, y_pred)

        # Update metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def gradient_step(self, x, y):
        """Compute multiple gradients per mesh"""
        total_loss = tf.constant(0.)

        # Average over subset of gradients which were computed for subsets of orientations
        idx = tf.constant(0)
        gradients = tf.TensorArray(
            tf.float32,
            size=self.splits,
            dynamic_size=False,
            clear_after_read=True,
            tensor_array_name="outer_ta",
            name="call_ta"
        )
        for rot in self.concurrent_rotations:
            with tf.GradientTape() as tape:
                y_pred = self(x, orientations=rot, training=True)
                loss = self.compute_loss(y=y, y_pred=y_pred)
            gradients.write(
                idx,
                tape.gradient(loss, self.trainable_variables)
            )
            total_loss = total_loss + loss
        gradients = tf.reduce_mean(gradients.stack(), axis=0)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return y_pred, total_loss
