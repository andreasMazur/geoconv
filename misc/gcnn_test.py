from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

from gcnn import ConvGeodesic
from dataset.MPI_FAUST.tf_dataset import load_preprocessed_faust

import tensorflow as tf
from datetime import datetime


class GeodesicNN(Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        (shot, bc), y = data

        with tf.GradientTape() as tape:
            y_pred = self((shot, bc), training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":

    # Define model
    signal_input = Input(shape=(6890, 1056))
    bary_input = Input(shape=(6890, 4, 2, 6))
    down_scaling = Dense(16, activation="relu")(signal_input)
    geodesic_conv = ConvGeodesic(
        kernel_size=(2, 4), output_dim=8, amt_kernel=2, activation="relu"
    )([down_scaling, bary_input])
    up_scaling = Dense(6890, activation="relu")(geodesic_conv)
    model = Model(inputs=[signal_input, bary_input], outputs=[up_scaling])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    model.summary()

    # Get data
    tf_faust_dataset = load_preprocessed_faust(
        "/home/andreas/PycharmProjects/Masterarbeit/dataset/MPI_FAUST/preprocessed_registrations.zip"
    )

    # Logs
    logdir = "./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=1, update_freq="epoch", write_steps_per_second=True, profile_batch=(1, 100)
    )

    # Model checkpoint
    checkpoint_path = "./training/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_freq='epoch', verbose=1)

    # Train
    model.fit(tf_faust_dataset.batch(1), epochs=2, callbacks=[tensorboard_callback, cp_callback])

    # Load saved model and used it
    model = tf.keras.models.load_model(checkpoint_path)
    tf_faust_dataset = load_preprocessed_faust(
        "/home/andreas/PycharmProjects/Masterarbeit/dataset/MPI_FAUST/preprocessed_registrations.zip"
    )
    for elem in tf_faust_dataset.batch(1):
        output = model(elem[0])
        print(output.shape)
