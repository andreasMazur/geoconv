from geoconv.tensorflow.backbone.imcnn_backbone import ImcnnBackbone
from geoconv.tensorflow.layers.barycentric_coordinates import BarycentricCoordinates
from geoconv_examples.modelnet_40_projections.dataset import load_preprocessed_modelnet

import os
import tensorflow as tf
import tensorflow_probability as tfp


class ModelnetClassifier(tf.keras.Model):
    def __init__(self,
                 n_radial,
                 n_angular,
                 n_neighbors,
                 template_scale,
                 adaption_data,
                 isc_layer_dims=None,
                 variant=None,
                 normalize=True):
        super().__init__()

        # BC-layer configuration
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.n_neighbors = n_neighbors
        self.template_scale = template_scale

        self.bc_layer = BarycentricCoordinates(
            self.n_radial,
            self.n_angular,
            n_neighbors=self.n_neighbors,
            template_scale=self.template_scale
        )
        self.bc_layer.trainable = False
        self.template_radius = self.bc_layer.adapt(data=adaption_data)

        # ISC-blocks configuration
        isc_layer_dims = [128, 64, 8] if isc_layer_dims is None else isc_layer_dims
        self.backbone = ImcnnBackbone(
            isc_layer_dims=isc_layer_dims,
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=self.template_radius,
            variant=variant,
            normalize=normalize
        )

        # Output configuration
        self.flatten = tf.keras.layers.Flatten()
        self.output_layer = tf.keras.layers.Dense(40)

    def call(self, inputs, **kwargs):
        # Estimate barycentric coordinates
        bc = self.bc_layer(inputs)

        # Embed
        signal = self.backbone([inputs, bc])

        # Compute covariance matrix from vertex-embeddings
        signal = self.flatten(tf.map_fn(tfp.stats.covariance, signal))

        # Classify covariance matrix
        return self.output_layer(signal)


def training(dataset_path,
             logging_dir,
             template_configurations=None,
             n_neighbors=20,
             variant=None,
             isc_layer_dims=None,
             learning_rate=0.00165):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    for (n_radial, n_angular, template_scale) in template_configurations:
        # Define and compile model
        imcnn = ModelnetClassifier(
            n_radial=n_radial,
            n_angular=n_angular,
            n_neighbors=n_neighbors,
            template_scale=template_scale,
            adaption_data=load_preprocessed_modelnet(dataset_path, is_train=True, batch_size=1),
            isc_layer_dims=isc_layer_dims,
            variant=variant
        )
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.AdamW(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=500,
                decay_rate=0.99
            ),
            weight_decay=0.005
        )
        imcnn.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        imcnn(tf.random.uniform(shape=[1, 2000, 3]))  # tf.TensorShape([None, 2000, 3])
        imcnn.summary()

        # Define callbacks
        exp_number = f"{n_radial}_{n_angular}_{template_scale}"
        csv_file_name = f"{logging_dir}/training_{exp_number}.log"
        csv = tf.keras.callbacks.CSVLogger(csv_file_name)
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, min_delta=0.01)
        tb = tf.keras.callbacks.TensorBoard(
            log_dir=f"{logging_dir}/tensorboard_{exp_number}",
            histogram_freq=1,
            write_graph=False,
            write_steps_per_second=True,
            update_freq="epoch",
            profile_batch=(1, 200)
        )

        # Load data
        train_data = load_preprocessed_modelnet(dataset_path, is_train=True)
        test_data = load_preprocessed_modelnet(dataset_path, is_train=False)

        # Train model
        imcnn.fit(x=train_data, callbacks=[stop, tb, csv], validation_data=test_data, epochs=200)
        imcnn.save(f"{logging_dir}/saved_imcnn_{exp_number}")
