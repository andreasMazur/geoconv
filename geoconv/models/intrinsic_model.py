from tensorflow import keras

import tensorflow as tf


class AbsoluteMean(keras.metrics.Metric):

    def __init__(self, name="absolute_mean", **kwargs):
        super(AbsoluteMean, self).__init__(name=name, **kwargs)
        self.absolute_mean = tf.Variable(
            tf.constant(0., dtype=tf.float32), name="absolute_mean_value", trainable=False
        )
        self.counter = tf.Variable(
            tf.constant(0., dtype=tf.float32), name="values_counter", trainable=False
        )

    def update_state(self, values):
        """Compute the absolute value and mean"""
        self.counter.assign_add(tf.constant(1., dtype=tf.float32))
        new_values = tf.math.reduce_mean(tf.math.abs(values))
        self.absolute_mean.assign((self.absolute_mean + new_values) / self.counter)

    def result(self):
        """Return the current absolute mean"""
        return self.absolute_mean


class CountGradients(keras.metrics.Metric):

    def __init__(self, name="gradient_counter", **kwargs):
        super(CountGradients, self).__init__(name=name, **kwargs)
        self.counted_gradients = tf.Variable(
            tf.constant(0, dtype=tf.int32), name="counted_gradients", trainable=False
        )

    def update_state(self, *args, **kwargs):
        """Update increments gradient counter by 1"""
        self.counted_gradients.assign_add(tf.constant(1, dtype=tf.int32))

    def result(self):
        """Return the current counter"""
        return self.counted_gradients


class ImCNN(keras.Model):

    def __init__(self, *args, splits=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.splits = splits

        # Capture gradient statistics
        self.gradient_stat_names = ["gradients_mean", "gradient_counter"]
        self.gradient_mean = AbsoluteMean(name=self.gradient_stat_names[0])
        self.gradient_counter = CountGradients(name=self.gradient_stat_names[1])

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
            elif metric.name in self.gradient_stat_names:
                pass
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def gradient_step(self, x, y):
        """Compute multiple gradients per mesh"""

        y = tf.stack(tf.split(y, self.splits))
        total_loss = tf.constant(0.)
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)
            original_shape = tf.shape(y_pred)

            y_pred = tf.stack(tf.split(y_pred, self.splits))
            for inner_idx in tf.range(self.splits):
                loss = self.compute_loss(y=y[inner_idx], y_pred=y_pred[inner_idx])

                with tape.stop_recording():
                    gradients = tape.gradient(loss, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                    # Update gradient dependent metrics
                    for g in gradients:
                        self.gradient_mean.update_state(g)
                    self.gradient_counter.update_state()
                    total_loss = total_loss + loss
        del tape
        return tf.reshape(y_pred, original_shape), total_loss
