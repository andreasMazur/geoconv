from geoconv_examples.modelnet_40.classifier import ModelNetClf
from geoconv_examples.modelnet_40.dataset import load_preprocessed_modelnet

import tensorflow as tf
import keras_tuner as kt
import os


def hyper_tuning(dataset_path,
                 logging_dir,
                 template_configuration,
                 neighbors_for_lrf,
                 modelnet10=True,
                 gen_info_file=None,
                 batch_size=1,
                 rotation_delta=1):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    n_radial, n_angular, template_radius = template_configuration

    def build_hypermodel(hp):
        # Get input
        signal_input = tf.keras.layers.Input(shape=(2000, 3), name="3D coordinates")

        # Configure classifier
        clf = ModelNetClf(
            neighbors_for_lrf=neighbors_for_lrf,
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=template_radius,
            isc_layer_dims=[100 for _ in range(4)],
            modelnet10=modelnet10,
            variant="dirac",
            rotation_delta=rotation_delta,
            dropout_rate=hp.Float("dropout_rate", min_value=0.01, max_value=0.5)
        )

        # Get signal embedding
        signal_output = clf(signal_input)

        # Compile model
        imcnn = tf.keras.Model(inputs=signal_input, outputs=signal_output)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.AdamW(
            learning_rate=hp.Float("learning_rate", min_value=0.00001, max_value=0.01),
            weight_decay=hp.Float("weight_decay", min_value=0.001, max_value=0.1)
        )
        imcnn.compile(optimizer=opt, loss=loss, metrics=["accuracy"], run_eagerly=True)

        return imcnn

    tuner = kt.Hyperband(
        hypermodel=build_hypermodel,
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
    tuner.search(x=train_data, validation_data=test_data, epochs=8, callbacks=[stop])

    # Print best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:")
    for key, value in best_hp.values.items():
        print(key, value)
