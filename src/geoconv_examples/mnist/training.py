from geoconv_examples.mnist.dataset import dataset

import tensorflow as tf
import keras_tuner as kt
import json


def hypertuning(mnist_atlas,
                n_radial,
                n_angular,
                batch_size,
                get_hypermodel,
                save_path,
                project_name,
                epochs=10,
                return_rotations=True):
    # Get data
    train_data, template_radius = dataset(
        mnist_atlas,
        set_type="train",
        n_radial=n_radial,
        n_angular=n_angular,
        batch_size=batch_size,
        return_rotations=return_rotations
    )
    test_data, _ = dataset(
        mnist_atlas,
        set_type="test",
        n_radial=n_radial,
        n_angular=n_angular,
        batch_size=batch_size,
        return_rotations=return_rotations
    )

    # Hyperparameter search
    tuner = kt.BayesianOptimization(
        hypermodel=get_hypermodel,
        objective=kt.Objective("val_sparse_categorical_accuracy", direction="max"),
        max_trials=1_000,
        num_initial_points=10,
        seed=42,
        project_name=project_name,
    )
    tuner.search(train_data, epochs=epochs, validation_data=test_data)

    # Save best model
    best_model = tuner.get_best_models()[0]
    if save_path[-6:] != ".keras":
        save_path += ".keras"
    best_model.save(save_path)


def training(mnist_atlas, n_radial, n_angular, batch_size, model, save_path, epochs=10, return_rotations=True):
    # Get data
    train_data, template_radius = dataset(
        mnist_atlas,
        set_type="train",
        n_radial=n_radial,
        n_angular=n_angular,
        batch_size=batch_size,
        return_rotations=return_rotations
    )
    test_data, _ = dataset(
        mnist_atlas,
        set_type="test",
        n_radial=n_radial,
        n_angular=n_angular,
        batch_size=batch_size,
        return_rotations=return_rotations
    )

    # Show model summary
    model.summary()

    # Train model
    term = tf.keras.callbacks.TerminateOnNaN()
    history = model.fit(x=train_data, batch_size=batch_size, epochs=epochs, validation_data=test_data, callbacks=[term])

    # Save model
    if save_path[-6:] != ".keras":
        save_path += ".keras"
    model.save(save_path)

    # Save history
    with open(f"{save_path[:-6]}_history.json", "w") as f:
        json.dump(history.history, f, indent=4)
