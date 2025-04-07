from geoconv.tensorflow.layers import ConvGeodesic
from geoconv.tensorflow.layers import ConvZero
from geoconv.utils.data_generator import read_template_configurations
from geoconv.utils.prepare_logs import process_logs
from geoconv_examples.mnist.tensorflow.dataset import load_preprocessed_mnist
from geoconv.tensorflow.layers import ConvDirac
from geoconv.tensorflow.layers import AngularMaxPooling

import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import os


class MNISTClassifier(keras.Model):
    def __init__(self, template_radius, variant=None, isc_layer_dims=None):
        super().__init__()

        if isc_layer_dims is None:
            isc_layer_dims = [128]

        if variant is None or variant == "dirac":
            self.layer_type = ConvDirac
        elif variant == "geodesic":
            self.layer_type = ConvGeodesic
        elif variant == "zero":
            self.layer_type = ConvZero
        else:
            raise RuntimeError("Select a layer type from: ['dirac', 'geodesic', 'zero']")

        self.convs = [
            self.layer_type(
                amt_templates=n,
                template_radius=template_radius,
                activation="elu",
                rotation_delta=1
            ) for n in isc_layer_dims
        ]
        self.amp = AngularMaxPooling()
        self.pool = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")
        self.output_layer = keras.layers.Dense(10, activation="linear")

    def call(self, inputs, **kwargs):
        signal, bc = inputs
        for layer in self.convs:
            signal = layer([signal, bc])
            signal = self.amp(signal)
        signal = self.pool(signal)
        return self.output_layer(signal)


def training(bc_path, logging_dir, k=5, template_configurations=None, variant=None, batch_size=8, isc_layer_dims=None):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Setup default layer parameterization if not given
    if isc_layer_dims is None:
        isc_layer_dims = [128]

    # Prepare k-fold cross-validation
    splits = tfds.even_splits("all", n=k)

    # Prepare template configurations
    if template_configurations is None:
        template_configurations = read_template_configurations(bc_path)

    # Run experiments
    for (n_radial, n_angular, template_radius) in template_configurations:
        csv_file_names = []
        for exp_no in range(len(splits)):
            # Load data
            train_data = load_preprocessed_mnist(
                bc_path,
                n_radial,
                n_angular,
                template_radius,
                set_type=splits[:exp_no] + splits[exp_no+1:],
                batch_size=batch_size
            )
            val_data = load_preprocessed_mnist(
                bc_path,
                n_radial,
                n_angular,
                template_radius,
                set_type=splits[exp_no],
                batch_size=batch_size
            )

            # Define and compile model
            imcnn = MNISTClassifier(template_radius, variant=variant, isc_layer_dims=isc_layer_dims)
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            imcnn.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

            # Define callbacks
            exp_number = f"{exp_no}__{n_radial}_{n_angular}_{template_radius}"
            csv_file_name = f"{logging_dir}/training_{exp_number}.log"
            csv_file_names.append(csv_file_name)
            csv = keras.callbacks.CSVLogger(csv_file_name)
            tb = keras.callbacks.TensorBoard(
                log_dir=f"{logging_dir}/tensorboard_{exp_number}",
                histogram_freq=1,
                write_graph=False,
                write_steps_per_second=True,
                update_freq="epoch",
                profile_batch=(1, 100)
            )
            save = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{logging_dir}/saved_imcnn_{exp_number}.keras",
                monitor="val_loss",
                save_best_only=True,
                save_freq="epoch"
            )

            # Train model
            imcnn.fit(x=train_data, callbacks=[tb, csv, save], validation_data=val_data, epochs=10)

        # Process logs
        process_logs(
            csv_file_names, file_name=f"{logging_dir}/avg_training_{n_radial}_{n_angular}_{template_radius}.log"
        )
