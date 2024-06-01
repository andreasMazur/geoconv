from geoconv.utils.common import read_template_configurations
from geoconv_examples.mnist.dataset import load_preprocessed_mnist
from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.angular_max_pooling import AngularMaxPooling

import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import os


class MNISTClassifier(keras.Model):
    def __init__(self, template_radius):
        super().__init__()
        self.conv = ConvDirac(
            amt_templates=128,
            template_radius=template_radius,
            activation="relu",
            rotation_delta=1
        )
        self.amp = AngularMaxPooling()
        self.output_layer = keras.layers.Dense(10)

    def call(self, inputs, **kwargs):
        signal, bc = [x[0] for x in inputs]
        signal = self.conv([signal, bc])
        signal = self.amp(signal)
        signal = tf.reshape(signal, (1, -1))
        return self.output_layer(signal)


def training(bc_path, logging_dir, k=5):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Prepare k-fold cross-validation
    splits = tfds.even_splits("all", n=k)

    # Prepare template configurations
    template_configurations = read_template_configurations(bc_path)

    # Run experiments
    exp_number = 0
    for (n_radial, n_angular, template_radius) in template_configurations:
        for exp_no in range(len(splits)):
            # Load data
            train_data = load_preprocessed_mnist(
                bc_path, n_radial, n_angular, template_radius, set_type=splits[:exp_no] + splits[exp_no+1:]
            )
            val_data = load_preprocessed_mnist(bc_path, n_radial, n_angular, template_radius, set_type=splits[exp_no])

            # Define and compile model
            imcnn = MNISTClassifier(template_radius)
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            imcnn.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

            # Define callbacks
            csv = keras.callbacks.CSVLogger(f"{logging_dir}/training_{exp_number}.log")
            stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
            tb = keras.callbacks.TensorBoard(
                log_dir=f"{logging_dir}/tensorboard_{exp_number}",
                histogram_freq=1,
                write_graph=False,
                write_steps_per_second=True,
                update_freq="epoch",
                profile_batch=(1, 70)
            )

            # Train model
            imcnn.fit(x=train_data, callbacks=[stop, tb, csv], validation_data=val_data, epochs=1)  # TODO epochs=200

            # Update experiment number
            exp_number += 1
            break
        break
