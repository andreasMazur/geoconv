from geoconv.tensorflow.layers import ConvDirac, ConvGeodesic, AngularMaxPooling
from geoconv.tensorflow.layers.activations.beta_relu import BetaRelu
from geoconv.tensorflow.layers.convolutions.conv_harmonic import ConvHarmonic
from geoconv.tensorflow.layers.descriptor.eucl_neighbors_descriptor import EuclNeighborsDescriptor
from geoconv.tensorflow.layers.experimental.lift_features import LiftFeatures2D
from geoconv.tensorflow.layers.convolutions.conv_gem import ConvGEM
from geoconv.tensorflow.layers.convolutions.conv_eman import ConvEMAN
from geoconv.tensorflow.layers.convolutions.conv_gem_p import ConvGEMP
from geoconv.tensorflow.layers.convolutions.conv_eman_p import ConvEMANP

from geoconv_examples.faust.dataset import dataset

import tensorflow as tf
import keras_tuner as kt
import json
import numpy as np
import os
import random


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
    """Runs hyperparameter tuning for a FAUST model and saves the best checkpoint.
        
    Parameters
    ----------
    faust_path: str
        The path to the preprocessed FAUST dataset.
    n_radial: int
        The amount of radial coordinates used by the discretized template.
    n_angular: int
        The amount of angular coordinates used by the discretized template.
    preprocess_method: str
        The charting algorithm.
    gpc_radius: float
        The max chart radius.
    template_radius: float
        The template radius.
    get_hypermodel: function
        The function that returns a hypermodel.
    save_path: str
        The path that points to the location where the best model shall be saved.
    project_name: str
        The name of the hypertuning run. It is used for logging purposes.
    epochs: int
        The amount of epochs per trial.
    return_rotations: bool
        Whether to enable return rotations for the network architecture.
    num_initial_points: int
        The number of initial configurations to test before starting to sample with acquisition function.
    max_trials: int
        The maximum number of trials.
    """
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

    # Hyperparameter search
    tuner = kt.BayesianOptimization(
        hypermodel=get_hypermodel,
        objective=kt.Objective("val_sparse_categorical_accuracy", direction="max"),
        max_trials=max_trials,
        num_initial_points=num_initial_points,
        seed=42,
        project_name=project_name,
    )
    term = tf.keras.callbacks.TerminateOnNaN()
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=0.001)
    tuner.search(train_data, epochs=epochs, validation_data=val_data, callbacks=[term, stop])

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
             return_rotations=True,
             random_seed=42,
             tensorboard_cb=False):
    """Trains- and benchmarks the trained model on the FAUST dataset.

    Parameters
    ----------
    faust_path: str
        The path to the preprocessed FAUST dataset.
    n_radial: int
        The amount of radial coordinates considered by the discretized template.
    n_angular: int
        The amount of angular coordinates considered by the discretized template.
    preprocess_method: str
        The used charting algorithm.
    gpc_radius: float
        The maximum allowed chart radius.
    template_radius: float
        The used template radius
    model: tf.keras.Model
        The neural network to train.
    save_path: str
        The path where model and benchmark stats will be saved.
    epochs: int
        The amount of training epochs.
    return_rotations: bool
        Whether the dataset should return rotations for parallel transport.
    random_seed: int
        A seed that initializes randomness.
    tensorboard_cb: bool
        Whether to use a tensorboard callback.
    """
    # Check if model already exists
    test_saving_path = f"{save_path}_test_history.json"
    if os.path.isfile(test_saving_path):
        print(f"{test_saving_path} already exists! Skipping training...")
        return

    # Set seeds
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

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
    if save_path[-6:] != ".keras":
        save_path += ".keras"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=True
    )
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10, min_delta=0.001)
    callbacks = [term, cp_callback, stop]

    if tensorboard_cb:
        tb_cb = tf.keras.callbacks.TensorBoard(
            log_dir=f"{os.path.dirname(save_path)}/tensorboard",
            histogram_freq=1,
            write_graph=False,
            write_steps_per_second=True,
            update_freq="epoch",
            profile_batch=(1, 700)
        )
        callbacks.append(tb_cb)

    train_history = model.fit(
        x=train_data, batch_size=1, epochs=epochs, validation_data=val_data, callbacks=callbacks
    )

    # Test best performing model
    model = tf.keras.models.load_model(
        save_path,
        custom_objects={
            "EuclNeighborsDescriptor": EuclNeighborsDescriptor,
            "ConvDirac": ConvDirac,
            "ConvGeodesic": ConvGeodesic,
            "AngularMaxPooling": AngularMaxPooling,
            "ConvHarmonic": ConvHarmonic,
            "BetaRelu": BetaRelu,
            "ConvGEM": ConvGEM,
            "ConvEMAN": ConvEMAN,
            "LiftFeatures2D": LiftFeatures2D,
            "ConvGEMP": ConvGEMP,
            "ConvEMANP": ConvEMANP,
        }
    )
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

    # Save history
    with open(f"{save_path[:-6]}_train_history.json", "w") as f:
        json.dump(train_history.history, f, indent=4)
    with open(test_saving_path, "w") as f:
        json.dump(test_history, f, indent=4)
