from tensorflow.keras import Model

import tensorflow as tf

BATCH_SIZE = 130


class PointCorrespondenceGeoCNN(Model):

    @tf.function
    def train_step(self, data):
        (signal, barycentric), ground_truth = data

        trainable_vars = self.trainable_variables
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self([signal, barycentric], training=True)
            loss = self.compiled_loss(ground_truth, y_pred)
            for batch_losses in tf.stack(tf.split(loss, BATCH_SIZE)):
                with tape.stop_recording():
                    gradients = tape.gradient(batch_losses, trainable_vars)
                    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(ground_truth, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        (signal, barycentric), ground_truth = data
        y_pred = self([signal, barycentric], training=False)
        self.compiled_loss(ground_truth, y_pred)
        self.compiled_metrics.update_state(ground_truth, y_pred)
        return {m.name: m.result() for m in self.metrics}
