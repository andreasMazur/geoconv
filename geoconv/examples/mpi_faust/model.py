from tensorflow.keras import Model

import tensorflow as tf

BATCH_SIZE = 689


class PointCorrespondenceGeoCNN(Model):

    # @tf.function
    def train_step(self, data):
        (signal, barycentric), ground_truth = data

        idx = tf.constant(0)
        losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self([signal, barycentric], training=True)
            all_losses = self.compiled_loss(ground_truth, y_pred) + tf.math.reduce_sum(self.losses)
            amt_losses = tf.shape(all_losses)[0]
            for i in tf.range(start=0, limit=amt_losses, delta=BATCH_SIZE):
                losses_batch = tf.reduce_sum(all_losses[i:i+BATCH_SIZE])
                tape.watch(losses_batch)
                losses = losses.write(idx, losses_batch)
                idx = idx + tf.constant(1)

        trainable_vars = self.trainable_variables
        for idx in tf.range(losses.size()):
            gradients = tape.gradient(losses.read(idx), trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # tape.reset()

        self.compiled_metrics.update_state(ground_truth, y_pred)
        return {m.name: m.result() for m in self.metrics}

