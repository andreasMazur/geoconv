from geoconv.tensorflow.layers.barycentric_coordinates import BarycentricCoordinates
from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.pooling.angular_max_pooling import AngularMaxPooling
from geoconv_examples.modelnet_40_projections.dataset import load_preprocessed_modelnet

import tensorflow as tf
import tensorflow_probability as tfp
import keras_tuner as kt
import os


class Covariance(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.map_fn(tfp.stats.covariance, inputs)


class HyperModel(kt.HyperModel):
    def __init__(self,
                 n_neighbors,
                 n_radial,
                 n_angular,
                 template_radius,
                 modelnet10=True):
        super().__init__()

        # Init barycentric coordinates layer
        self.bc_layer = BarycentricCoordinates(
            n_radial=n_radial,
            n_angular=n_angular,
            n_neighbors=n_neighbors,
            template_scale=None
        )
        self.bc_layer.adapt(template_radius=template_radius)

        # Init angular max-pooling layer
        self.amp = AngularMaxPooling()

        # Remember template radius
        self.template_radius = template_radius

        # Remember dataset type
        self.modelnet10 = modelnet10

    def build(self, hp):
        # Get input
        signal_input = tf.keras.layers.Input(shape=(2000, 3), name="3D coordinates")

        # Compute barycentric coordinates
        bc = self.bc_layer(signal_input)

        # Compute vertex embeddings
        embedding = signal_input
        for idx in range(3):
            embedding = ConvDirac(
                amt_templates=hp.Int(name=f"ISC_layer_{idx}", min_value=8, max_value=32),
                template_radius=self.template_radius,
                activation="relu",
                name=f"ISC_layer_{idx}",
                rotation_delta=1
            )([embedding, bc])
            embedding = self.amp(embedding)

        # Compute flat covariance matrices
        embedding = Covariance()(embedding)
        embedding = tf.keras.layers.Flatten()(embedding)

        # Compute output logits
        signal_output = tf.keras.layers.Dense(10 if self.modelnet10 else 40)(embedding)

        # Compile model
        imcnn = tf.keras.Model(inputs=signal_input, outputs=signal_output)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.AdamW(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=hp.Float("initial_learning_rate", min_value=0.0005, max_value=0.005),
                decay_steps=500,
                decay_rate=0.99
            ),
            weight_decay=0.005
        )
        imcnn.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

        return imcnn


def hyper_tuning(dataset_path,
                 logging_dir,
                 template_configuration,
                 n_neighbors,
                 modelnet10=True,
                 gen_info_file=None,
                 batch_size=1):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    n_radial, n_angular, template_radius = template_configuration
    tuner = kt.Hyperband(
        hypermodel=HyperModel(
            n_neighbors,
            n_radial,
            n_angular,
            template_radius,
            modelnet10
        ),
        objective="val_accuracy",
        max_epochs=200,
        factor=3,
        directory=logging_dir,
        project_name="modelnet_40_hyper_tuning"
    )

    # Setup datasets
    train_data = load_preprocessed_modelnet(
        dataset_path,
        is_train=True,
        modelnet10=modelnet10,
        gen_info_file=f"{logging_dir}/{gen_info_file}",
        batch_size=batch_size
    )
    test_data = load_preprocessed_modelnet(
        dataset_path,
        is_train=False,
        modelnet10=modelnet10,
        gen_info_file=f"{logging_dir}/test_{gen_info_file}",
        batch_size=batch_size
    )

    # Start hyperparameter tuning
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=0.01)
    tuner.search(x=train_data, validation_data=test_data, epochs=200, callbacks=[stop])

    # Print best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:")
    for key, value in best_hp.values.items():
        print(key, value)
