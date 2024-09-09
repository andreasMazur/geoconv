from geoconv.tensorflow.layers.barycentric_coordinates import BarycentricCoordinates
from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.pooling.angular_max_pooling import AngularMaxPooling
from geoconv_examples.modelnet_40.classifier import ModelNetClf
from geoconv_examples.modelnet_40.dataset import load_preprocessed_modelnet

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
        self.modelnet_clf = ModelNetClf(
            n_neighbors=n_neighbors,
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=template_radius,
            isc_layer_dims=[64, 128, 256],
            modelnet10=modelnet10,
            variant="dirac",
            rotation_delta=4
        )

    def build(self, hp):
        # Get input
        signal_input = tf.keras.layers.Input(shape=(2000, 3), name="3D coordinates")

        # Get signal embedding
        signal_output = self.modelnet_clf(signal_input)

        # Compile model
        imcnn = tf.keras.Model(inputs=signal_input, outputs=signal_output)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.AdamW(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=hp.Float("initial_learning_rate", min_value=0.00001, max_value=0.001),
                decay_steps=500,
                decay_rate=0.99
            ),
            weight_decay=0.005
        )
        imcnn.compile(optimizer=opt, loss=loss, metrics=["accuracy"], run_eagerly=True)

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
        set_type="train",
        modelnet10=modelnet10,
        gen_info_file=f"{logging_dir}/{gen_info_file}",
        batch_size=batch_size
    )
    test_data = load_preprocessed_modelnet(
        dataset_path,
        set_type="test",
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
