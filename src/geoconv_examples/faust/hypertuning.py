from geoconv_examples.faust.dataset import load_preprocessed_faust
from geoconv_examples.faust.training import SIG_DIM, AMOUNT_VERTICES, FaustVertexClassifier

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
        self.normalize.adapt(adapt_data)

    def build(self, hp):
        # Define model input
        signal_input = tf.keras.layers.Input(shape=(AMOUNT_VERTICES, SIG_DIM), name="Signal")
        bc_input = tf.keras.layers.Input(shape=(AMOUNT_VERTICES, self.n_radial, self.n_angular, 3, 2), name="BC")

        # Normalize input
        signal = self.normalize(signal_input)

        # Predict vertex embeddings
        vertex_predictions = FaustVertexClassifier(
            self.template_radius,
            isc_layer_dims=[256, 128, 64, 32, 16],
            middle_layer_dim=64,
            variant="dirac",
            normalize_input=False,
            rotation_delta=1,
            dropout_rate=hp.Float("dropout_rate", min_value=0.01, max_value=0.9),
            output_rotation_delta=1,
            l1_reg=hp.Float("l1_reg_coefficient", min_value=0.00001, max_value=0.001),
            initializer="glorot_uniform"
        )([signal, bc_input])

        # Compile model
        imcnn = tf.keras.Model(inputs=[signal_input, bc_input], outputs=vertex_predictions)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.AdamW(
            learning_rate=hp.Float("initial_learning_rate", min_value=0.00001, max_value=0.01),
            weight_decay=0.005
        )
        imcnn.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        imcnn.build(
            input_shape=[
                tf.TensorShape([None, AMOUNT_VERTICES, 3]),
                tf.TensorShape([None, AMOUNT_VERTICES, self.n_radial, self.n_angular, 3, 2])
            ]
        )

        return imcnn


def hyper_tuning(dataset_path, logging_dir, template_configuration, gen_info_file=None):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Set filename for generator
    if gen_info_file is None:
        gen_info_file = "generator_info.json"

    # Load datasets
    n_radial, n_angular, template_radius = template_configuration
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

    # Initialize hypermodel
    hyper_model = HyperModel(
        n_radial,
        n_angular,
        template_radius,
        load_preprocessed_faust(
            dataset_path,
            n_radial,
            n_angular,
            template_radius,
            is_train=True,
            gen_info_file=f"{logging_dir}/{gen_info_file}",
            only_signal=True
        )
    )

    # Run hyper-tuning
    tuner = kt.Hyperband(
        hypermodel=hyper_model,
        objective=kt.Objective("val_accuracy", direction="max"),
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
