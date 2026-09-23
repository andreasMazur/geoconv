from geoconv_examples.mnist.dataset import dataset

import tensorflow as tf
import keras_tuner as kt
import json


def hypertuning(mnist_atlas,
                n_radial,
                n_angular,
                template_radius,
                batch_size,
                get_hypermodel,
                save_path,
                project_name,
                epochs=10,
                return_rotations=True):
    """Runs hyperparameter tuning for the MNIST surface models.

    Parameters
    ----------
    mnist_atlas: Atlas
        The mnist atlas.
    n_radial: int
        The amount of radial coordinates used by the discretized template.
    n_angular: int
        The amount of angular coordinates used by the discretized template.
    template_radius: float
        The template radius.
    batch_size: int
        The batch size.
    get_hypermodel: function
        A function creating a hypermodel.
    save_path: str
        The path that points to the location where the best model shall be saved.
    project_name: str
        The project name. It is used for logging purposes.
    epochs: int
        The number epochs for a single trial.
    return_rotations: bool
        Whether to enable return rotations.
    """
    # Get data
    train_data = dataset(
        mnist_atlas,
        set_type="train",
        n_radial=n_radial,
        n_angular=n_angular,
        radius=template_radius,
        batch_size=batch_size,
        return_rotations=return_rotations
    )
    test_data = dataset(
        mnist_atlas,
        set_type="test",
        n_radial=n_radial,
        n_angular=n_angular,
        radius=template_radius,
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


def training(mnist_atlas,
             n_radial,
             n_angular,
             template_radius,
             batch_size,
             model,
             save_path,
             epochs=10,
             return_rotations=True):
    """Training a surface CNN on MNIST.

    Parameters
    ----------
    mnist_atlas: Atlas
        An Atlas for the MNIST images.
    n_radial: int
        The amount of radial coordinates used by the discretized template.
    n_angular: int
        The amount of angular coordinates used by the discretized template.
    template_radius: float
        The template radius
    batch_size: int
        The batch size
    model: tf.keras.Model
        The surface CNN to be trained.
    save_path: str
        The path to save the trained model and benchmark stats.
    epochs: int
        The number of training epochs.
    return_rotations: bool
        Whether the dataset returns rotations for a parallel transport.
    """
    # Get data
    train_data = dataset(
        mnist_atlas,
        set_type="train",
        n_radial=n_radial,
        n_angular=n_angular,
        radius=template_radius,
        batch_size=batch_size,
        return_rotations=return_rotations
    )
    test_data = dataset(
        mnist_atlas,
        set_type="test",
        n_radial=n_radial,
        n_angular=n_angular,
        radius=template_radius,
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
