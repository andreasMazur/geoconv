from tensorflow.keras import Model

import tensorflow as tf


class PointCorrespondenceGeoCNN(Model):

    def train_step(self, data):
        (signal, barycentric), ground_truth = data

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self([signal, barycentric], training=True)
            losses = self.compiled_loss(ground_truth, y_pred) + self.losses  # Add regularization loss
            losses_list = []
            for loss in losses:
                tape.watch(loss)
                losses_list.append(loss)

        trainable_vars = self.trainable_variables
        for loss in losses_list:
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        tape.reset()

        self.compiled_metrics.update_state(ground_truth, y_pred)
        return {m.name: m.result() for m in self.metrics}

