from geoconv_examples.faust.dataset import dataset

import tensorflow as tf
import keras_tuner as kt
import json
import os


def hypertuning(faust_path,
                n_radial,
                n_angular,
                preprocess_method,
                gpc_radius,
                template_radius,
                get_hypermodel,
                save_path,
                project_name,
                epochs=10,
                return_rotations=True,
                num_initial_points=10,
                max_trials=100):
    # Get data
    train_data = dataset(
        zip_path=faust_path,
        set_type="train",
        n_radial=n_radial,
        n_angular=n_angular,
        preprocess_method=preprocess_method,
        gpc_radius=gpc_radius,
        template_radius=tf.constant(template_radius, dtype=tf.float64),
        return_rotations=return_rotations
    )
    val_data = dataset(
        zip_path=faust_path,
        set_type="validation",
        n_radial=n_radial,
        n_angular=n_angular,
        preprocess_method=preprocess_method,
        gpc_radius=gpc_radius,
        template_radius=tf.constant(template_radius, dtype=tf.float64),
        return_rotations=return_rotations
    )

    # Hyperparameter search
    tuner = kt.BayesianOptimization(
        hypermodel=get_hypermodel,
        objective=kt.Objective("val_sparse_categorical_accuracy", direction="max"),
        max_trials=max_trials,
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


def training(faust_path,
             n_radial,
             n_angular,
             preprocess_method,
             gpc_radius,
             template_radius,
             model,
             save_path,
             epochs=10,
             return_rotations=True):
    # Get data
    train_data = dataset(
        zip_path=faust_path,
        set_type="train",
        n_radial=n_radial,
        n_angular=n_angular,
        preprocess_method=preprocess_method,
        gpc_radius=tf.constant(gpc_radius, dtype=tf.float64),
        template_radius=tf.constant(template_radius, dtype=tf.float64),
        return_rotations=return_rotations
    )
    val_data = dataset(
        zip_path=faust_path,
        set_type="validation",
        n_radial=n_radial,
        n_angular=n_angular,
        preprocess_method=preprocess_method,
        gpc_radius=tf.constant(gpc_radius, dtype=tf.float64),
        template_radius=tf.constant(template_radius, dtype=tf.float64),
        return_rotations=return_rotations
    )

    # Show model summary
    model.summary()

    # Train model
    term = tf.keras.callbacks.TerminateOnNaN()
    train_history = model.fit(x=train_data, batch_size=1, epochs=epochs, validation_data=val_data, callbacks=[term])

    # Test model
    test_data = dataset(
        zip_path=faust_path,
        set_type="test",
        n_radial=n_radial,
        n_angular=n_angular,
        preprocess_method=preprocess_method,
        gpc_radius=tf.constant(gpc_radius, dtype=tf.float64),
        template_radius=tf.constant(template_radius, dtype=tf.float64),
        return_rotations=return_rotations
    )
    test_history = model.evaluate(test_data, return_dict=True)

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path[-6:] != ".keras":
        save_path += ".keras"
    model.save(save_path)

    # Save history
    with open(f"{save_path[:-6]}_train_history.json", "w") as f:
        json.dump(train_history.history, f, indent=4)
    with open(f"{save_path[:-6]}_test_history.json", "w") as f:
        json.dump(test_history, f, indent=4)
