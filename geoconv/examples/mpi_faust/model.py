from tensorflow.keras import Model

import tensorflow as tf

AMT_TRAINING_SPLITS = 5


class PointCorrespondenceGeoCNN(Model):

    def train_step(self, data):
        (signal, barycentric), ground_truth = data
        barycentric = tf.split(barycentric, AMT_TRAINING_SPLITS)
        ground_truth = tf.split(ground_truth, AMT_TRAINING_SPLITS)

        for bc, gt in zip(barycentric, ground_truth):
            with tf.GradientTape() as tape:
                y_pred = self([signal, bc], training=True)
                loss = self.compiled_loss(gt, y_pred, regularization_losses=self.losses)

            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(ground_truth, y_pred)
        return {m.name: m.result() for m in self.metrics}

