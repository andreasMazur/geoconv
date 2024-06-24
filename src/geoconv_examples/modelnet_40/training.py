from geoconv.tensorflow.backbone.imcnn_backbone import ImcnnBackbone
from geoconv.utils.data_generator import read_template_configurations
from geoconv_examples.modelnet_40.dataset import load_preprocessed_modelnet

import os
import keras
import tensorflow as tf
import tensorflow_probability as tfp


class ModelnetClassifier(keras.Model):
    def __init__(self, n_radial, n_angular, template_radius, isc_layer_dims=None, variant=None, normalize=True):
        super().__init__()
        isc_layer_dims = [128, 64, 8] if isc_layer_dims is None else isc_layer_dims
        self.backbone = ImcnnBackbone(
            isc_layer_dims=isc_layer_dims,
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=template_radius,
            variant=variant,
            normalize=normalize
        )
        self.flatten = tf.keras.layers.Flatten()
        self.output_layer = tf.keras.layers.Dense(40)

    def call(self, inputs, **kwargs):
        # Embed
        signal = self.backbone(inputs)
        # Flatten embeddings
        signal = self.flatten(tf.map_fn(tfp.stats.covariance, signal))
        # Output
        return self.output_layer(signal)


def training(dataset_path, logging_dir, template_configurations=None, variant=None):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Prepare template configurations
    if template_configurations is None:
        template_configurations = read_template_configurations(dataset_path)

    # Run experiments
    for (n_radial, n_angular, template_radius) in template_configurations:
        # Load data
        train_data = load_preprocessed_modelnet(dataset_path, n_radial, n_angular, template_radius, is_train=True)
        test_data = load_preprocessed_modelnet(dataset_path, n_radial, n_angular, template_radius, is_train=False)

        # Define and compile model
        imcnn = ModelnetClassifier(n_radial, n_angular, template_radius, variant=variant)
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
        imcnn.build(
            input_shape=[tf.TensorShape([None, 541, 3]), tf.TensorShape([None, 541, n_radial, n_angular, 3, 2])]
        )
        print("Adapt normalization layer on training data..")
        imcnn.backbone.normalize.adapt(
            load_preprocessed_modelnet(
                dataset_path, n_radial, n_angular, template_radius, is_train=True, only_signal=True
            )
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
            profile_batch=(1, 200)
        )

        # Train model
        imcnn.fit(x=train_data, callbacks=[stop, tb, csv], validation_data=test_data, epochs=200)
        imcnn.save(f"{logging_dir}/saved_imcnn_{exp_number}")
