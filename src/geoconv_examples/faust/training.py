from geoconv.tensorflow.layers.pooling.angular_max_pooling import AngularMaxPooling
from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.conv_geodesic import ConvGeodesic
from geoconv.tensorflow.layers.conv_zero import ConvZero
from geoconv.utils.data_generator import read_template_configurations
from geoconv.utils.princeton_benchmark import princeton_benchmark
from geoconv_examples.faust.dataset import FAUST_FOLDS, load_preprocessed_faust

import tensorflow as tf
import keras
import os


class FaustModel(keras.Model):
    def __init__(self, n_radial, n_angular, template_radius, variant=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_dim = 544
        self.kernel_size = (n_radial, n_angular)
        self.template_radius = template_radius
        self.output_dims = [96, 256, 384, 384]
        self.rotation_deltas = [1 for _ in range(len(self.output_dims))]

        if variant is None or variant == "dirac":
            self.layer_type = ConvDirac
        elif variant == "geodesic":
            self.layer_type = ConvGeodesic
        elif variant == "zero":
            self.layer_type = ConvZero
        else:
            raise RuntimeError("Select a layer type from: ['dirac', 'geodesic', 'zero']")

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
                self.layer_type(
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


def training(bc_path, logging_dir, reference_mesh_path, template_configurations=None, variant=None, processes=1):
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
            imcnn = FaustModel(n_radial, n_angular, template_radius, variant=variant)
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            opt = keras.optimizers.AdamW(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.00165,
                    decay_steps=500,
                    decay_rate=0.99
                ),
                weight_decay=0.005
            )
            imcnn.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

            # Adapt normalization on training data
            print("Initializing normalization layer..")
            imcnn.normalize.build(tf.TensorShape([6890, 544]))
            adaption_data = load_preprocessed_faust(
                bc_path, n_radial, n_angular, template_radius, is_train=True, split=exp_no, only_signal=True
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
            stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, min_delta=0.01)
            tb = keras.callbacks.TensorBoard(
                log_dir=f"{logging_dir}/tensorboard_{exp_number}",
                histogram_freq=1,
                write_graph=False,
                write_steps_per_second=True,
                update_freq="epoch",
                profile_batch=(1, 80)
            )

            # Train model
            imcnn.fit(x=train_data, callbacks=[stop, tb, csv], validation_data=test_data, epochs=200)
            imcnn.save(f"{logging_dir}/saved_imcnn_{exp_number}")

            # Evaluate model with Princeton benchmark
            test_data = load_preprocessed_faust(
                bc_path, n_radial, n_angular, template_radius, is_train=False, split=exp_no
            )
            princeton_benchmark(
                imcnn=imcnn,
                test_dataset=test_data,
                ref_mesh_path=reference_mesh_path,
                normalize=True,
                file_name=f"{logging_dir}/model_benchmark_{exp_number}",
                processes=processes,
                geodesic_diameter=2.2093810817030244
            )
