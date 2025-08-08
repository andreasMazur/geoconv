from geoconv_examples.faust.tensorflow.classifier import build_faust_classifier
from geoconv_examples.faust.tensorflow.dataset import load_preprocessed_faust

import tensorflow as tf
import keras_tuner as kt
import os


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
        gen_info_file=f"{logging_dir}/{gen_info_file}",
        batch_size=1,
    )
    test_data = load_preprocessed_faust(
        dataset_path,
        n_radial,
        n_angular,
        template_radius,
        is_train=False,
        gen_info_file=f"{logging_dir}/test_{gen_info_file}",
        batch_size=1,
    )

    def build_hypermodel(hp):
        imcnn = build_faust_classifier(
            variant=hp.Choice(name="kernel", values=["dirac", "geodesic"]),
            n_radial=n_radial,
            n_angular=n_angular,
            isc_layer_dims=[
                hp.Int(name="isc_layer_1", min_value=8, max_value=16, step=8),
                hp.Int(name="isc_layer_2", min_value=8, max_value=16, step=8),
            ],
            template_radius=template_radius,
            rotation_delta=train_data.element_spec[0][1].shape[3],
        )
        imcnn.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float(name="learning_rate", min_value=1e-6, max_value=0.1)
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        return imcnn

    tuner = kt.BayesianOptimization(
        hypermodel=build_hypermodel,
        objective=kt.Objective(name="val_accuracy", direction="max"),
        max_trials=10_000,
        num_initial_points=12,
        directory=logging_dir,
        project_name="mnist_hyper_tuning",
        tune_new_entries=True,
        allow_new_entries=True
    )

    # Start hyperparameter tuning
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=0.01)
    tuner.search(x=train_data, validation_data=test_data, epochs=200, callbacks=[stop])

    # Print best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:")
    for key, value in best_hp.values.items():
        print(key, value)
