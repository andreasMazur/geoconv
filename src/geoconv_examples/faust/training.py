from geoconv_examples.faust.dataset import dataset, adapt_generator

import tensorflow as tf
import keras_tuner as kt
import json


def hypertuning(faust_path,
                n_radial,
                n_angular,
                radius,
                get_hypermodel,
                save_path,
                project_name,
                epochs=10,
                return_rotations=True,
                num_initial_points=100):
    # Get data
    train_data = dataset(
        zip_path=faust_path,
        set_type="train",
        n_radial=n_radial,
        n_angular=n_angular,
        radius=tf.constant(radius, dtype=tf.float64),
        return_rotations=return_rotations
    )
    val_data = dataset(
        zip_path=faust_path,
        set_type="validation",
        n_radial=n_radial,
        n_angular=n_angular,
        radius=tf.constant(radius, dtype=tf.float64),
        return_rotations=return_rotations
    )

    # Hyperparameter search
    tuner = kt.BayesianOptimization(
        hypermodel=get_hypermodel,
        objective=kt.Objective("val_sparse_categorical_accuracy", direction="max"),
        max_trials=1_000,
        num_initial_points=num_initial_points,
        seed=42,
        project_name=project_name,
    )
    tuner.search(train_data, epochs=epochs, validation_data=val_data)

    # Save best model
    best_model = tuner.get_best_models()[0]
    if save_path[-6:] != ".keras":
        save_path += ".keras"
    best_model.save(save_path)


def training(faust_path, n_radial, n_angular, radius, batch_size, model, save_path, epochs=10, return_rotations=True):
    # Get data
    train_data = dataset(
        zip_path=faust_path,
        set_type="train",
        n_radial=n_radial,
        n_angular=n_angular,
        radius=tf.constant(radius, dtype=tf.float64),
        return_rotations=return_rotations
    )
    val_data = dataset(
        zip_path=faust_path,
        set_type="validation",
        n_radial=n_radial,
        n_angular=n_angular,
        radius=tf.constant(radius, dtype=tf.float64),
        return_rotations=return_rotations
    )

    # Show model summary
    model.summary()

    # Adapt normalization layer
    normalization_layer = [l for l in model.layers if "normalization" == l.name][0]
    descr_layer = [l for l in model.layers if "descr" in l.name][0]
    normalization_layer.adapt(
        adapt_generator(faust_path, "train", n_radial, n_angular, radius, descr_layer)
    )

    # Train model
    term = tf.keras.callbacks.TerminateOnNaN()
    history = model.fit(x=train_data, batch_size=batch_size, epochs=epochs, validation_data=val_data, callbacks=[term])

    # Save model
    if save_path[-6:] != ".keras":
        save_path += ".keras"
    model.save(save_path)

    # Save history
    with open(f"{save_path[:-6]}_history.json", "w") as f:
        json.dump(history.history, f, indent=4)
