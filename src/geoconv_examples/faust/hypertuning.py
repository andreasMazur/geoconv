from geoconv_examples.faust.dataset import load_preprocessed_faust
from geoconv_examples.faust.training import SIG_DIM, FaustVertexClassifier, reconstruction_loss

import tensorflow as tf
import keras_tuner as kt
import os


class HyperModel(kt.HyperModel):
    def __init__(self, n_radial, n_angular, template_radius, adapt_data):
        super().__init__()
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.template_radius = template_radius

        self.normalize = tf.keras.layers.Normalization(axis=-1, name="input_normalization")
        print("Adapt normalization layer on training data..")
        self.normalize.adapt(adapt_data)
        print("Done.")

    def build(self, hp):
        # Define model input
        signal_input = tf.keras.layers.Input(shape=(SIG_DIM,), name="Signal")
        bc_input = tf.keras.layers.Input(shape=(6890, self.n_radial, self.n_angular, 3, 2), name="BC")

        # Normalize input
        signal = self.normalize(signal_input)

        vertex_predictions, reconstructions = FaustVertexClassifier(
            self.n_radial,
            self.n_angular,
            self.template_radius,
            isc_layer_dims=[
                hp.Int(name="ISC_1", min_value=201, max_value=300),
                hp.Int(name="ISC_2", min_value=101, max_value=200),
                hp.Int(name="ISC_3", min_value=1, max_value=100),
            ],
            normalize=False
        )([signal, bc_input])

        # Compile model
        imcnn = tf.keras.Model(inputs=[signal_input, bc_input], outputs=[vertex_predictions, reconstructions])
        losses = [
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            reconstruction_loss
        ]
        opt = tf.keras.optimizers.AdamW(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=hp.Float("initial_learning_rate", min_value=0.001, max_value=0.005),
                decay_steps=500,
                decay_rate=0.99
            ),
            weight_decay=0.005
        )
        imcnn.compile(optimizer=opt, loss=losses, metrics=["accuracy"])
        imcnn.build(
            input_shape=[
                tf.TensorShape([None, 6890, 3]), tf.TensorShape([None, 6890, self.n_radial, self.n_angular, 3, 2])
            ]
        )

        return imcnn


def hyper_tuning(dataset_path, logging_dir, template_configuration):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Run hyper-tuning
    n_radial, n_angular, template_radius = template_configuration
    train_data = load_preprocessed_faust(dataset_path, n_radial, n_angular, template_radius, is_train=True)
    test_data = load_preprocessed_faust(dataset_path, n_radial, n_angular, template_radius, is_train=False)
    adapt_data = load_preprocessed_faust(
        dataset_path, n_radial, n_angular, template_radius, is_train=True, only_signal=True
    )

    tuner = kt.Hyperband(
        hypermodel=HyperModel(n_radial, n_angular, template_radius, adapt_data),
        objective="val_accuracy",
        max_epochs=200,
        factor=3,
        directory=logging_dir,
        project_name="faust_hyper_tuning"
    )

    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=0.01)
    tuner.search(x=train_data, validation_data=test_data, epochs=200, callbacks=[stop])

    # Print best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:")
    for key, value in best_hp.values.items():
        print(key, value)
