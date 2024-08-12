from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.conv_geodesic import ConvGeodesic
from geoconv.tensorflow.layers.pooling.angular_max_pooling import AngularMaxPooling
from geoconv.utils.data_generator import read_template_configurations
from geoconv.utils.princeton_benchmark import princeton_benchmark
from geoconv_examples.faust.dataset import load_preprocessed_faust

import tensorflow as tf
import keras
import os


SIG_DIM = 544


def reconstruction_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.abs(y_pred))


class FaustVertexClassifier(keras.Model):
    def __init__(self,
                 template_radius,
                 isc_layer_dims=None,
                 variant=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Determine which layer type shall be used
        variant = "dirac" if variant is None else variant
        if variant not in ["dirac", "geodesic"]:
            raise RuntimeError(
                f"'{variant}' is not a valid network type. Please select a valid variant from ['dirac', 'geodesic']."
            )

        # Init ISC block
        self.isc_layers = []
        for idx in range(len(isc_layer_dims)):
            if variant == "dirac":
                self.isc_layers.append(
                    ConvDirac(
                        amt_templates=isc_layer_dims[idx],
                        template_radius=template_radius,
                        activation="relu",
                        name=f"ISC_layer_{idx}",
                        rotation_delta=1
                    )
                )
            else:
                self.isc_layers.append(
                    ConvGeodesic(
                        amt_templates=isc_layer_dims[idx],
                        template_radius=template_radius,
                        activation="relu",
                        name=f"ISC_layer_{idx}",
                        rotation_delta=1
                    )
                )
        self.amp = AngularMaxPooling()

        self.output_dense = keras.layers.Dense(6890, name="output")

    def call(self, inputs, **kwargs):
        signal, bc = inputs

        # Compute vertex embeddings
        for idx in range(len(self.isc_layers)):
            signal = self.isc_layers[idx]([signal, bc])
            signal = self.amp(signal)

        # Output
        return self.output_dense(signal)


def training(dataset_path,
             logging_dir,
             reference_mesh_path,
             template_configurations=None,
             variant=None,
             processes=1,
             isc_layer_dims=None,
             learning_rate=0.00165,
             gen_info_file=None):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Prepare template configurations
    if template_configurations is None:
        template_configurations = read_template_configurations(dataset_path)

    # Set filename for generator
    if gen_info_file is None:
        gen_info_file = "generator_info.json"

    # Run experiments
    for (n_radial, n_angular, template_radius) in template_configurations:

        # Load data
        train_data = load_preprocessed_faust(
            dataset_path,
            n_radial,
            n_angular,
            template_radius,
            is_train=True,
            gen_info_file=f"{logging_dir}/{gen_info_file}"
        )
        test_data = load_preprocessed_faust(
            dataset_path,
            n_radial,
            n_angular,
            template_radius,
            is_train=False,
            gen_info_file=f"{logging_dir}/test_{gen_info_file}"
        )

        # Define and compile model
        imcnn = FaustVertexClassifier(template_radius, isc_layer_dims=isc_layer_dims, variant=variant)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = keras.optimizers.AdamW(
            learning_rate=keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=500,
                decay_rate=0.99
            ),
            weight_decay=0.005
        )
        imcnn.compile(optimizer=opt, loss=[loss, reconstruction_loss], metrics=["accuracy"])
        imcnn.build(
            input_shape=[tf.TensorShape([None, 6890, SIG_DIM]), tf.TensorShape([None, 6890, n_radial, n_angular, 3, 2])]
        )
        imcnn.summary()

        # Define callbacks
        exp_number = f"{n_radial}_{n_angular}_{template_radius}"
        csv_file_name = f"{logging_dir}/training_{exp_number}.log"
        csv = keras.callbacks.CSVLogger(csv_file_name)
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
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
            dataset_path,
            n_radial,
            n_angular,
            template_radius,
            is_train=False,
            gen_info_file=f"{logging_dir}/test_{gen_info_file}"
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
