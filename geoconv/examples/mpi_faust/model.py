from tensorflow.keras import Model

import tensorflow as tf

BATCH_SIZE = 130


class PointCorrespondenceGeoCNN(Model):

    @tf.function
    def train_step(self, data):
        (signal, barycentric), ground_truth = data

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self([signal, barycentric], training=True)
            loss = self.compiled_loss(ground_truth, y_pred)
            # Batch losses
            loss = tf.stack(tf.split(loss, BATCH_SIZE))
            loss = tf.math.reduce_sum(loss, axis=1) / tf.cast(tf.shape(loss)[1], tf.float32)

        trainable_vars = self.trainable_variables
        gradients = tape.jacobian(loss, trainable_vars, experimental_use_pfor=False)
        for i in tf.range(BATCH_SIZE):
            self.optimizer.apply_gradients(zip([g[i] for g in gradients], trainable_vars))

        self.compiled_metrics.update_state(ground_truth, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        (signal, barycentric), ground_truth = data
        y_pred = self([signal, barycentric], training=False)
        self.compiled_loss(ground_truth, y_pred)
        self.compiled_metrics.update_state(ground_truth, y_pred)
        return {m.name: m.result() for m in self.metrics}
