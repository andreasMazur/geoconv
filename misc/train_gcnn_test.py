from tensorflow.keras.layers import Input
from tensorflow.keras import Model

from dataset.tf_MPI_FAUST_dataset import load_preprocessed_faust
from gcnn import ConvGeodesic

import tensorflow as tf
import datetime


class GCNN(Model):
    def train_step(self, data):
        shot, bc, label_matrix = data

        with tf.GradientTape() as tape:
            y_pred = self([shot, bc], training=True)
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(label_matrix, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(label_matrix, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def define_model(signal_shape, bc_shape):
    signal_input = Input(shape=signal_shape, name="signal")
    bary_input = Input(shape=bc_shape, name="Barycentric c.")
    geodesic_conv = ConvGeodesic(
        kernel_size=(2, 4), output_dim=1, amt_kernel=1, activation="relu"
    )([signal_input, bary_input])
    model = GCNN(inputs=[signal_input, bary_input], outputs=[geodesic_conv])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train():
    tf_faust_dataset = load_preprocessed_faust("/home/andreas/Uni/Masterarbeit/MPI-FAUST/preprocessed_faust.zip")
    model = define_model(signal_shape=(6890, 1), bc_shape=(6890, 8, 8))

    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(tf_faust_dataset, epochs=1, callbacks=[tensorboard_callback])


if __name__ == "__main__":
    train()
