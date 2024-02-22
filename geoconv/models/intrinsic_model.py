from tensorflow import keras

import tensorflow as tf


class ImCNN(keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    def test_step(self, data):
        (signal, bc), gt = data
        pred = self([signal, bc], training=False)
        loss = self.compute_loss(y=gt, y_pred=pred)

        # Statistics
        self.loss_tracker.update_state(loss)
        for metric in self.metrics[1:]:  # skip loss
            metric.update_state(gt, pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        (signal, bc), gt = data

        with tf.GradientTape() as tape:
            pred = self([signal, bc], training=True)
            loss = self.compute_loss(y=gt, y_pred=pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Statistics
        self.loss_tracker.update_state(loss)
        for metric in self.metrics[1:]:  # skip loss
            metric.update_state(gt, pred)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # 'reset_states()' called automatically at the start of each training epoch or evaluation
        return [self.loss_tracker, self.accuracy]
