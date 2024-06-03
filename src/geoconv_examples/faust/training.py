from geoconv.tensorflow.layers.angular_max_pooling import AngularMaxPooling
from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.utils.common import read_template_configurations
from geoconv_examples.faust.dataset import FAUST_FOLDS, load_preprocessed_faust

import tensorflow as tf
import keras
import os


class FaustModel(keras.Model):
    def __init__(self, n_radial, n_angular, template_radius, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_dim = 544
        self.kernel_size = (n_radial, n_angular)
        self.template_radius = template_radius
        self.output_dims = [96, 256, 384, 384]
        self.rotation_deltas = [1 for _ in range(len(self.output_dims))]

        #################
        # Handling Input
        #################
        self.normalize = keras.layers.Normalization(axis=-1, name="input_normalization")
        self.downsize_dense = keras.layers.Dense(64, activation="relu", name="downsize")
        self.downsize_bn = keras.layers.BatchNormalization(axis=-1, name="BN_downsize")

        #############
        # ISC blocks
        #############
        self.isc_layers = []
        self.bn_layers = []
        self.do_layers = []
        self.amp_layers = []
        for idx in range(len(self.output_dims)):
            self.do_layers.append(keras.layers.Dropout(rate=0.2, name=f"DO_layer_{idx}"))
            self.isc_layers.append(
                ConvDirac(
                    amt_templates=self.output_dims[idx],
                    template_radius=self.template_radius,
                    activation="relu",
                    name=f"ISC_layer_{idx}",
                    rotation_delta=self.rotation_deltas[idx]
                )
            )
            self.bn_layers.append(keras.layers.BatchNormalization(axis=-1, name=f"BN_layer_{idx}"))
            self.amp_layers.append(AngularMaxPooling())

        #########
        # Output
        #########
        self.output_dense = keras.layers.Dense(6890, name="output")

    def call(self, inputs, **kwargs):
        #################
        # Handling Input
        #################
        signal, bc = inputs
        signal = self.normalize(signal)
        signal = self.downsize_dense(signal)
        signal = self.downsize_bn(signal)

        ###############
        # Forward pass
        ###############
        for idx in range(len(self.output_dims)):
            signal = self.do_layers[idx](signal)
            signal = self.isc_layers[idx]([signal, bc])
            signal = self.amp_layers[idx](signal)
            signal = self.bn_layers[idx](signal)

        #########
        # Output
        #########
        return self.output_dense(signal)


def training(bc_path, logging_dir, template_configurations=None):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Prepare template configurations
    if template_configurations is None:
        template_configurations = read_template_configurations(bc_path)

    # Run experiments
    for (n_radial, n_angular, template_radius) in template_configurations:
        for exp_no in range(len(FAUST_FOLDS.keys())):
            # Load data
            train_data = load_preprocessed_faust(
                bc_path, n_radial, n_angular, template_radius, is_train=True, split=exp_no
            )
            test_data = load_preprocessed_faust(
                bc_path, n_radial, n_angular, template_radius, is_train=False, split=exp_no
            )

            # Define and compile model
            imcnn = FaustModel(n_radial, n_angular, template_radius)
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            imcnn.compile(optimizer="adam", loss=loss, metrics=["accuracy"], run_eagerly=True)

            # Adapt normalization
            print("Initializing normalization layer..")
            imcnn.normalize.build(tf.TensorShape([6890, 544]))
            adaption_data = load_preprocessed_faust(
                bc_path, n_radial, n_angular, template_radius, is_train=False, split=exp_no, only_signal=True
            )
            imcnn.normalize.adapt(adaption_data)
            print("Done.")

            # Build model
            imcnn([
                tf.random.uniform(shape=(6890, 544)), tf.random.uniform(shape=(6890,) + (n_radial, n_angular) + (3, 2))
            ])
            imcnn.summary()

            # Define callbacks
            exp_number = f"{exp_no}__{n_radial}_{n_angular}_{template_radius}"
            csv = keras.callbacks.CSVLogger(f"{logging_dir}/training_{exp_number}.log")
            tb = keras.callbacks.TensorBoard(
                log_dir=f"{logging_dir}/tensorboard_{exp_number}",
                histogram_freq=1,
                write_graph=False,
                write_steps_per_second=True,
                update_freq="epoch",
                profile_batch=(1, 80)
            )

            # Train model
            imcnn.fit(x=train_data, callbacks=[tb, csv], validation_data=test_data, epochs=200)
