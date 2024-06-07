from geoconv.tensorflow.layers.angular_max_pooling import AngularMaxPooling
from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.conv_geodesic import ConvGeodesic
from geoconv.tensorflow.layers.conv_zero import ConvZero
from geoconv.utils.data_generator import read_template_configurations
from geoconv_examples.modelnet_40.dataset import load_preprocessed_modelnet, MODELNET40_FOLDS

import os
import keras
import tensorflow as tf


class ModelnetClassifier(keras.Model):
    def __init__(self, template_radius, variant=None):
        super().__init__()

        if variant is None or variant == "dirac":
            self.layer_type = ConvDirac
        elif variant == "geodesic":
            self.layer_type = ConvGeodesic
        elif variant == "zero":
            self.layer_type = ConvZero
        else:
            raise RuntimeError("Select a layer type from: ['dirac', 'geodesic', 'zero']")

        self.conv_1 = self.layer_type(
            amt_templates=128,
            template_radius=template_radius,
            activation="relu",
            rotation_delta=1
        )
        self.conv_2 = self.layer_type(
            amt_templates=128,
            template_radius=template_radius,
            activation="relu",
            rotation_delta=1
        )
        self.conv_3 = self.layer_type(
            amt_templates=128,
            template_radius=template_radius,
            activation="relu",
            rotation_delta=1
        )

        self.amp = AngularMaxPooling()
        self.global_avg = keras.layers.GlobalAveragePooling1D()
        self.output_layer = keras.layers.Dense(40)

    def call(self, inputs, **kwargs):
        signal, bc = inputs

        # Embed
        signal = self.conv_1([signal, bc])
        signal = self.amp(signal)
        signal = self.conv_2([signal, bc])
        signal = self.amp(signal)
        signal = self.conv_3([signal, bc])
        signal = self.amp(signal)

        # Average pool
        signal = self.global_avg(tf.reshape(signal, (1, -1, 128)))

        # Output
        return self.output_layer(signal)


def training(bc_path, logging_dir, template_configurations=None, variant=None):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Prepare template configurations
    if template_configurations is None:
        template_configurations = read_template_configurations(bc_path)

    # Run experiments
    for (n_radial, n_angular, template_radius) in template_configurations:
        for exp_no in range(len(MODELNET40_FOLDS.keys())):
            # Load data
            train_data = load_preprocessed_modelnet(
                bc_path, n_radial, n_angular, template_radius, is_train=True, split=exp_no
            )
            test_data = load_preprocessed_modelnet(
                bc_path, n_radial, n_angular, template_radius, is_train=False, split=exp_no
            )

            # Define and compile model
            imcnn = ModelnetClassifier(template_radius, variant=variant)
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            imcnn.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

            # Define callbacks
            exp_number = f"{exp_no}__{n_radial}_{n_angular}_{template_radius}"
            csv = keras.callbacks.CSVLogger(f"{logging_dir}/training_{exp_number}.log")
            stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, min_delta=0.01)
            tb = keras.callbacks.TensorBoard(
                log_dir=f"{logging_dir}/tensorboard_{exp_number}",
                histogram_freq=1,
                write_graph=False,
                write_steps_per_second=True,
                update_freq="epoch",
                profile_batch=(1, 100)
            )

            # Train model
            imcnn.fit(x=train_data, callbacks=[stop, tb, csv], validation_data=test_data, epochs=200)
            imcnn.save(f"{logging_dir}/saved_imcnn_{exp_number}")
