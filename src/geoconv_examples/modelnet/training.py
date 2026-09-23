from geoconv.tensorflow.layers import ConvDirac, ConvGeodesic, AngularMaxPooling
from geoconv.tensorflow.layers.activations.beta_relu import BetaRelu
from geoconv.tensorflow.layers.convolutions.conv_harmonic import ConvHarmonic
from geoconv.tensorflow.layers.descriptor.eucl_neighbors_descriptor import EuclNeighborsDescriptor
from geoconv.tensorflow.layers.experimental.lift_features import LiftFeatures2D
from geoconv.tensorflow.layers.convolutions.conv_gem import ConvGEM
from geoconv.tensorflow.layers.convolutions.conv_eman import ConvEMAN
from geoconv.tensorflow.layers.convolutions.conv_gem_p import ConvGEMP
from geoconv.tensorflow.layers.convolutions.conv_eman_p import ConvEMANP
from geoconv.tensorflow.layers.experimental.unlift_features import UnLiftFeatures2D
from geoconv_examples.modelnet.architectures.deep_sets import DeepSet
from geoconv_examples.modelnet.dataset import dataset

import tensorflow as tf
import keras_tuner as kt
import json
import numpy as np
import os
import random


def hypertuning(mn10_path,
                n_radial,
                n_angular,
                preprocess_method,
                gpc_radius,
                get_hypermodel,
                save_path,
                project_name,
                epochs=10,
                return_rotations=True,
                num_initial_points=10,
                max_trials=100):
    """Runs hyperparameter tuning for a ModelNet10 surface CNN.

    Parameters
    ----------
    mn10_path: str
        The path that points to the location of the preprocessed ModelNet10 dataset.
    n_radial: int
        The number of radial coordinates of the template.
    n_angular: int
        The number of angular coordinates of the template.
    preprocess_method: str
        The charting method.
    gpc_radius: float
        The maximum chart radius.
    get_hypermodel: function
        The function that returns hypermodel for hyperparameter tuning.
    save_path: str
        The path that points to the location where the best model shall be saved.
    project_name: str
        The project name. Used for logging purposes.
    epochs: int
        The maximum number of epochs for one trial.
    return_rotations: bool
        Whether the dataset should return rotation angles and whether the architecture expects parallel transport
        angles.
    num_initial_points: int
        The number of randomly picked initial trials before starting to use the acquisition function.
    max_trials: int
        The maximum number of trials.
    """
    # Get data
    train_data = dataset(
        path=mn10_path,
        set_type="train",
        n_radial=n_radial,
        n_angular=n_angular,
        chart_max_radius=tf.constant(gpc_radius, dtype=tf.float64),
        method=preprocess_method,
        return_rotations=return_rotations,
        do_zero_pad=True
    )
    val_data = dataset(
        path=mn10_path,
        set_type="val",
        n_radial=n_radial,
        n_angular=n_angular,
        chart_max_radius=tf.constant(gpc_radius, dtype=tf.float64),
        method=preprocess_method,
        return_rotations=return_rotations,
        do_zero_pad=True
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


def training(mn10_path,
             n_radial,
             n_angular,
             preprocess_method,
             gpc_radius,
             model,
             save_path,
             epochs=10,
             random_seed=42,
             return_rotations=True,
             tensorboard_cb=False):
    """Starts the training for a given surface CNN on ModelNet10.

    Parameters
    ----------
    mn10_path: str
        The path to the preprocessed ModelNet10 dataset.
    n_radial: int
        The amount of radial coordinates considered by the discretized template.
    n_angular: int
        The amount of angular coordinates considered by the discretized template.
    preprocess_method: str
        The used charting algorithm.
    gpc_radius: float
        The maximum allowed chart radius.
    model: tf.Keras.Model
        The surface CNN to be trained.
    save_path: str
        The path to where the trained surface CNN and benchmark statistics shall be stored.
    epochs: int
        The number of training epochs.
    random_seed: int
        A seed to initialize randomness.
    return_rotations: bool
        Whether the dataset shall return angles for parallel transports.
    tensorboard_cb: bool
        Whether to include a tensorboard callback during training.
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
        path=mn10_path,
        set_type="train",
        n_radial=n_radial,
        n_angular=n_angular,
        chart_max_radius=tf.constant(gpc_radius, dtype=tf.float64),
        method=preprocess_method,
        return_rotations=return_rotations,
        do_zero_pad=True
    )
    val_data = dataset(
        path=mn10_path,
        set_type="val",
        n_radial=n_radial,
        n_angular=n_angular,
        chart_max_radius=tf.constant(gpc_radius, dtype=tf.float64),
        method=preprocess_method,
        return_rotations=return_rotations,
        do_zero_pad=True
    )

    # Show model summary
    model.summary()

    # Train model
    term = tf.keras.callbacks.TerminateOnNaN()
    if save_path[-6:] != ".keras":
        save_path += ".keras"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cp_callback_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=True
    )
    save_path_acc = f"{save_path[:-6]}_accuracy.keras"
    cp_callback_acc = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path_acc,
        monitor="val_sparse_categorical_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=True
    )
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10, min_delta=0.001)
    callbacks = [term, cp_callback_loss, cp_callback_acc, stop]

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

    train_history = model.fit(x=train_data, batch_size=1, epochs=epochs, validation_data=val_data, callbacks=callbacks)

    # Save history
    with open(f"{save_path[:-6]}_train_history.json", "w") as f:
        json.dump(train_history.history, f, indent=4)

    # Test best performing model
    for idx, sp in enumerate([save_path, save_path_acc]):
        model = tf.keras.models.load_model(
            sp,
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
                "DeepSet": DeepSet,
                "UnLiftFeatures2D": UnLiftFeatures2D,
            }
        )
        test_data = dataset(
            path=mn10_path,
            set_type="test",
            n_radial=n_radial,
            n_angular=n_angular,
            chart_max_radius=tf.constant(gpc_radius, dtype=tf.float64),
            method=preprocess_method,
            return_rotations=return_rotations,
            do_zero_pad=True
        )
        test_history = model.evaluate(test_data, return_dict=True)

        if idx == 0:
            with open(test_saving_path, "w") as f:
                json.dump(test_history, f, indent=4)
        else:
            with open(f"{save_path[:-6]}_test_history_accuracy.json", "w") as f:
                json.dump(test_history, f, indent=4)
