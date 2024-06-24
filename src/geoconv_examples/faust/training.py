from geoconv.tensorflow.backbone.imcnn_backbone import ImcnnBackbone
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
                 n_radial,
                 n_angular,
                 template_radius,
                 isc_layer_dims=None,
                 variant=None,
                 normalize=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        isc_layer_dims = [128, 64, 8] if isc_layer_dims is None else isc_layer_dims
        self.backbone_encoder = ImcnnBackbone(
            isc_layer_dims=isc_layer_dims,
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=template_radius,
            variant=variant,
            normalize=normalize
        )
        self.backbone_decoder = ImcnnBackbone(
            isc_layer_dims=isc_layer_dims[1::-1] + [SIG_DIM],
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=template_radius,
            variant=variant,
            normalize=False
        )
        self.output_dense = keras.layers.Dense(6890, name="output")

    def call(self, inputs, **kwargs):
        signal, bc = inputs

        # Embed
        embedding = self.backbone_encoder([signal, bc])

        # Reconstruct
        reconstruction = self.backbone_decoder([embedding, bc])

        # Output
        return self.output_dense(embedding), signal - reconstruction


def training(dataset_path,
             logging_dir,
             reference_mesh_path,
             template_configurations=None,
             variant=None,
             processes=1):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Prepare template configurations
    if template_configurations is None:
        template_configurations = read_template_configurations(dataset_path)

    # Run experiments
    for (n_radial, n_angular, template_radius) in template_configurations:

        # Load data
        train_data = load_preprocessed_faust(dataset_path, n_radial, n_angular, template_radius, is_train=True)
        test_data = load_preprocessed_faust(dataset_path, n_radial, n_angular, template_radius, is_train=False)

        # Define and compile model
        imcnn = FaustVertexClassifier(n_radial, n_angular, template_radius, variant=variant)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = keras.optimizers.AdamW(
            learning_rate=keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.00165,
                decay_steps=500,
                decay_rate=0.99
            ),
            weight_decay=0.005
        )
        imcnn.compile(optimizer=opt, loss=[loss, reconstruction_loss], metrics=["accuracy"])
        imcnn.build(
            input_shape=[tf.TensorShape([None, 6890, SIG_DIM]), tf.TensorShape([None, 6890, n_radial, n_angular, 3, 2])]
        )
        print("Adapt normalization layer on training data..")
        imcnn.backbone_encoder.normalize.adapt(
            load_preprocessed_faust(dataset_path, n_radial, n_angular, template_radius, is_train=True, only_signal=True)
        )
        print("Done.")
        imcnn.summary()

        # Define callbacks
        exp_number = f"{n_radial}_{n_angular}_{template_radius}"
        csv_file_name = f"{logging_dir}/training_{exp_number}.log"
        csv = keras.callbacks.CSVLogger(csv_file_name)
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
        test_data = load_preprocessed_faust(dataset_path, n_radial, n_angular, template_radius, is_train=False)
        princeton_benchmark(
            imcnn=imcnn,
            test_dataset=test_data,
            ref_mesh_path=reference_mesh_path,
            normalize=True,
            file_name=f"{logging_dir}/model_benchmark_{exp_number}",
            processes=processes,
            geodesic_diameter=2.2093810817030244
        )
