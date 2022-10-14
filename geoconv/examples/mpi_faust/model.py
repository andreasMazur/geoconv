from tensorflow.keras import Model

import tensorflow as tf

BATCH_SIZE = 689


class PointCorrespondenceGeoCNN(Model):

    @tf.function
    def train_step(self, data):
        (signal, barycentric), ground_truth = data

        trainable_vars = self.trainable_variables
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self([signal, barycentric], training=True)
            all_losses = self.compiled_loss(ground_truth, y_pred) + tf.math.reduce_sum(self.losses)
            amt_losses = tf.shape(all_losses)[0]
            for i in tf.range(start=0, limit=amt_losses, delta=BATCH_SIZE):
                losses_batch = tf.reduce_sum(all_losses[i:i+BATCH_SIZE])
                tape.watch(losses_batch)
                gradients = tape.gradient(losses_batch, trainable_vars)
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(ground_truth, y_pred)
        return {m.name: m.result() for m in self.metrics}

