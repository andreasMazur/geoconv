from geoconv_examples.mnist.tensorflow.dataset import load_preprocessed_mnist
from geoconv_examples.mnist.tensorflow.training import MNISTClassifier, build_mnist_classifier

import os
import keras_tuner as kt
import tensorflow as tf


def hyper_tuning(dataset_path,
                 logging_dir,
                 n_radial,
                 n_angular,
                 template_radius,
                 batch_size):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Load data
    train_data = load_preprocessed_mnist(
        dataset_path,
        n_radial,
        n_angular,
        template_radius,
        set_type="train",
        batch_size=batch_size
    )
    test_data = load_preprocessed_mnist(
        dataset_path,
        n_radial,
        n_angular,
        template_radius,
        set_type="test",
        batch_size=batch_size
    )

    def build_hypermodel(hp):
        imcnn = build_mnist_classifier(
            variant=hp.Choice(name="kernel", values=["dirac", "geodesic"]),
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=template_radius,
            rotation_delta=train_data.element_spec[0][1].shape[3],
            isc_layer_dims=[
                hp.Int(name="isc_layer_1", min_value=8, max_value=16, step=8),
                hp.Int(name="isc_layer_2", min_value=8, max_value=16, step=8),
            ]
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
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, min_delta=0.0001)
    tuner.search(x=train_data, validation_data=test_data, epochs=5, callbacks=[stop])

    # Print best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:")
    for key, value in best_hp.values.items():
        print(key, value)
