from geoconv.tensorflow.backbone.imcnn_backbone import ImcnnBackbone
from geoconv.tensorflow.layers.barycentric_coordinates import BarycentricCoordinates
from geoconv_examples.modelnet_40_projections.dataset import load_preprocessed_modelnet

import tensorflow_probability as tfp
import tensorflow as tf
import keras_tuner as kt
import os


class Covariance(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.map_fn(tfp.stats.covariance, inputs)


class HyperModel(kt.HyperModel):
    def __init__(self, n_radial, n_angular, template_radius, adapt_data, n_neighbors):
        super().__init__()
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.template_radius = template_radius
        self.adapt_data = adapt_data

        self.n_neighbors = n_neighbors

        # Barycentric coordinates layer
        self.bc_layer = BarycentricCoordinates(
            self.n_radial,
            self.n_angular,
            n_neighbors=self.n_neighbors,
            template_scale=None  # Initialize directly with template radius for now
        )
        self.bc_layer.trainable = False
        self.template_radius = self.bc_layer.adapt(template_radius=template_radius)

        # Normalization layer
        self.normalize = tf.keras.layers.Normalization(axis=-1, name="input_normalization")
        self.normalize.adapt(adapt_data)

    def build(self, hp):
        # Define model input
        signal_input = tf.keras.layers.Input(shape=(2000, 3), name="Signal")

        # Get BC
        bc = self.bc_layer(signal_input)

        # Normalize input
        signal = self.normalize(signal_input)

        # Embed
        signal = ImcnnBackbone(
            isc_layer_dims=[
                hp.Int(name="ISC_1", min_value=96, max_value=400),
                hp.Int(name="ISC_2", min_value=96, max_value=400),
                hp.Int(name="ISC_3", min_value=96, max_value=400),
                hp.Int(name="ISC_4", min_value=96, max_value=400),
            ],
            n_radial=self.n_radial,
            n_angular=self.n_angular,
            template_radius=self.template_radius,
            variant="dirac",
            normalize=False
        )([signal, bc])

        # Output
        signal = Covariance()(signal)
        signal = tf.keras.layers.Flatten()(signal)
        signal_output = tf.keras.layers.Dense(10)(signal)  # Work on modelnet10 for now

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
        imcnn.build(
            input_shape=[
                tf.TensorShape([None, 541, 3]), tf.TensorShape([None, 541, self.n_radial, self.n_angular, 3, 2])
            ]
        )

        return imcnn


def hyper_tuning(dataset_path, logging_dir, template_configuration, gen_info_file):
    gen_info_file_1 = f"{logging_dir}/{gen_info_file}"
    gen_info_file_2 = f"{logging_dir}/test_{gen_info_file}"

    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Run hyper-tuning
    n_radial, n_angular, template_radius = template_configuration
    train_data = load_preprocessed_modelnet(dataset_path, is_train=True, modelnet10=True, gen_info_file=gen_info_file_1)
    test_data = load_preprocessed_modelnet(dataset_path, is_train=False, modelnet10=True, gen_info_file=gen_info_file_2)
    adapt_data = load_preprocessed_modelnet(
        dataset_path, is_train=True, only_signal=True, modelnet10=True, gen_info_file=gen_info_file_1
    )

    tuner = kt.Hyperband(
        hypermodel=HyperModel(n_radial, n_angular, template_radius, adapt_data, n_neighbors=10),
        objective="val_accuracy",
        max_epochs=200,
        factor=3,
        directory=logging_dir,
        project_name="modelnet_40_hyper_tuning"
    )

    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=0.01)
    tuner.search(x=train_data, validation_data=test_data, epochs=200, callbacks=[stop])

    # Print best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:")
    for key, value in best_hp.values.items():
        print(key, value)
