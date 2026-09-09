from geoconv.tensorflow.layers import ConvDirac, ConvGeodesic, AngularMaxPooling
from geoconv.tensorflow.layers.activations.beta_relu import BetaRelu
from geoconv.tensorflow.layers.convolutions.conv_harmonic import ConvHarmonic
from geoconv.tensorflow.layers.convolutions.conv_gem import ConvGEM
from geoconv.tensorflow.layers.convolutions.conv_eman import ConvEMAN
from geoconv.tensorflow.layers.convolutions.conv_gem_p import ConvGEMP
from geoconv.tensorflow.layers.convolutions.conv_eman_p import ConvEMANP
from geoconv_examples.planetswe.dataset import dataset
from geoconv_examples.planetswe.training_configs.dictionaries import PLANETSWE_NORM_VALUES
from geoconv_examples.planetswe.vrmse import compute_vrmse

import os
import tensorflow as tf
import numpy as np
import random
import json
import keras_tuner as kt


def hypertuning(bc_path,
                swe_path,
                batch_size,
                return_rotations,
                add_input_zero_dim,
                get_hypermodel,
                max_trials,
                num_initial_points,
                project_name,
                save_path,
                predict_residual):
    """Runs hyperparameter tuning for the PlanetSWE models and stores the best model.

    Parameters
    ----------
    bc_path: str
        The path that points to the location where the barycentric coordinates for the sphere are saved.
    swe_path: str
        The path that points to the downloaded PlanetSWE dataset. Used to retrieve sphere signals.
    batch_size: int
        The batch size.
    return_rotations: bool
        Whether the dataset should return rotation angles and whether the architecture expects parallel transport
        angles.
    add_input_zero_dim: bool
        Whether the dataset should concatenate a zero dimension to the height field such that returned signal vectors
        are 4-dimensional. Gauge-equivariant architecture expect even-dimensional input vectors.
    get_hypermodel: function
        The function that returns a hypermodel for hyperparameter tuning.
    max_trials: int
        The maximum amount of trials used during Bayesian optimization.
    num_initial_points: int
        The number of randomly picked initial trials before starting to use the acquisition function.
    project_name: str
        The project name. Used for logging purposes
    save_path: str
        The path that points to the location where the best model shall be saved.
    predict_residual: bool
        Whether the model should predict residuals which are added onto the current state instead of the full state for
        the next time step.
    """
    # Get data
    train_data = dataset(
        bc_path=bc_path,
        swe_path=swe_path,
        set_type="train",
        batch_size=batch_size,
        return_rotations=return_rotations,
        add_input_zero_dim=add_input_zero_dim,
        return_differences=predict_residual
    )
    val_data = dataset(
        bc_path=bc_path,
        swe_path=swe_path,
        set_type="valid",
        batch_size=batch_size,
        return_rotations=return_rotations,
        add_input_zero_dim=add_input_zero_dim,
        return_differences=predict_residual
    )

    # Hyperparameter search
    tuner = kt.BayesianOptimization(
        hypermodel=get_hypermodel,
        objective=kt.Objective("val_vrmse", direction="min"),
        max_trials=max_trials,
        num_initial_points=num_initial_points,
        seed=42,
        project_name=project_name,
    )
    term = tf.keras.callbacks.TerminateOnNaN()
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=0.001)
    tuner.search(train_data, epochs=1, validation_data=val_data, callbacks=[term, stop])

    # Save best model
    best_model = tuner.get_best_models()[0]
    save_path = save_path if save_path.endswith(".keras") else f"{save_path}.keras"
    best_model.save(save_path)


def rollout_benchmark(model, test_data, t_max, save_path, zero_pad=False):
    """Computes the rollout error for a trajectory up to time-step 't_max'.

    Parameters
    ----------
    model: tf.keras.Model
        The model to benchmark.
    test_data: tf.data.Dataset
        The test dataset.
    t_max: int
        The maximum time-step to test for.
    save_path: str
        The path pointing to where to save the rollout errors.
    zero_pad: bool
        Whether the dataset should concatenate a zero dimension to the height field such that returned signal vectors
        are 4-dimensional. Gauge-equivariant architecture expect even-dimensional input vectors.
    """
    # Load normalization values
    normalization_means = np.array(PLANETSWE_NORM_VALUES["channel_means"])
    normalization_stds = np.array(PLANETSWE_NORM_VALUES["channel_stds"])

    time_step_errors = []
    prediction = None
    for (time_step, t_feature_field, barycentric_coordinates), t_next_feature_field in test_data:
        # Set preceding prediction
        if time_step == 0:
            preceding_prediction = t_feature_field
            print("Next trajectory..")
        else:
            preceding_prediction = prediction

        # Estimate next time step from preceding prediction
        prediction = model([preceding_prediction, barycentric_coordinates], training=False)

        # De-normalize values for VRMSE metric
        de_normalized_prediction = prediction * normalization_stds + normalization_means
        t_next_feature_field = t_next_feature_field * normalization_stds + normalization_means
        vrmse_t_next = compute_vrmse(t_next_feature_field, de_normalized_prediction, axis=(1, 2))

        # Remember VRMSE
        time_step_errors.append(vrmse_t_next)

        # Console logging
        print(f"\rt: {time_step % t_max} -> t+1 {(time_step % t_max) + 1}: VRMSE(t+1) = {vrmse_t_next}")

        # Zero pad prediction if required for architecture
        if zero_pad:
            prediction = tf.concat([prediction, tf.zeros(tf.shape(prediction)[:2])[..., None]], axis=-1)

    # Save time-step errors
    time_step_errors = np.array(time_step_errors).reshape(-1, t_max)
    np.save(save_path, time_step_errors)


def benchmark(model, test_data, save_path, stride=1):
    """Computes the VRMSE for one-step predictions.

    Parameters
    ----------
    model: tf.keras.Model
        The model to benchmark.
    test_data: tf.data.Dataset
        The test dataset.
    save_path: str
        The path pointing to where to save the rollout errors.
    stride: int
        A stride for the time step. If 'stride=1', then all time steps are predicted. If 'stride=n', then
        merely every n-th time step will be predicted.
    """
    # Load normalization values
    normalization_means = np.array(PLANETSWE_NORM_VALUES["channel_means"])
    normalization_stds = np.array(PLANETSWE_NORM_VALUES["channel_stds"])

    time_step_errors = []
    for (time_step, t_feature_field, barycentric_coordinates), t_next_feature_field in test_data:
        if int(time_step) % stride != 0:
            continue

        # Estimate next time step from preceding prediction
        prediction = model([t_feature_field, barycentric_coordinates], training=False)

        # De-normalize values for VRMSE metric
        de_normalized_prediction = prediction * normalization_stds + normalization_means
        t_next_feature_field = t_next_feature_field * normalization_stds + normalization_means
        vrmse_t_next = compute_vrmse(t_next_feature_field, de_normalized_prediction, axis=(1, 2))

        # Remember VRMSE
        time_step_errors.append(vrmse_t_next)

        # Console logging
        print(f"\rt: {time_step} -> t+1 {time_step + 1}: VRMSE(t+1) = {vrmse_t_next}")

    # Save time-step errors
    time_step_errors = np.array(time_step_errors).reshape(4, -1)
    np.save(save_path, time_step_errors)


def training(model,
             bc_path,
             swe_path,
             return_rotations,
             save_path,
             random_seed=42,
             tensorboard_cb=False,
             batch_size=1,
             add_input_zero_dim=False,
             predict_residual=False):
    """Trains a PlanetSWE model, checkpoints the best run, and stores training statistics.

    Parameters
    ----------
    model: tf.keras.Model
        The model to operate on.
    bc_path: str
        The path that points to the location where the barycentric coordinates for the sphere are saved.
    swe_path: str
        The path that points to the downloaded PlanetSWE dataset. Used to retrieve sphere signals.
    return_rotations: bool
        Whether the dataset should return rotation angles and whether the architecture expects parallel transport
        angles.
    save_path: str
        The path that points to the location where the trained model shall be saved.
    random_seed: int
        The random seed used throughout the training.
    tensorboard_cb: bool
        Whether to enable the tensorboard callback.
    batch_size: int
        The batch size.
    add_input_zero_dim: Any
        Whether the dataset should concatenate a zero dimension to the height field such that returned signal vectors
        are 4-dimensional. Gauge-equivariant architecture expect even-dimensional input vectors.
    predict_residual: bool
        Whether the model should predict residuals which are added onto the current state instead of the full state for
        the next time step.
    """
    ######################
    # Define saving paths
    ######################
    os.makedirs(save_path, exist_ok=True)
    tensorboard_callback_path = f"{save_path}/tensorboard"
    cp_callback_loss_path = f"{save_path}/loss_callback.keras"
    cp_callback_metric_path = f"{save_path}/metric_callback.keras"
    rollout_statistics_loss_path = f"{save_path}/loss_rollout_statistics.json"
    rollout_statistics_metric_path = f"{save_path}/metric_rollout_statistics.json"
    training_logs_json = f"{save_path}/training_logs.json"

    ##########################################################################################
    # Check if metric rollout already has been computed - last benchmark in training function
    ##########################################################################################
    if os.path.isfile(rollout_statistics_metric_path):
        print(f"{rollout_statistics_metric_path} already exists! Skipping training...")
        return

    ############
    # Set seeds
    ############
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    ###########
    # Get data
    ###########
    train_data = dataset(
        bc_path=bc_path,
        swe_path=swe_path,
        set_type="train",
        batch_size=batch_size,
        return_rotations=return_rotations,
        add_input_zero_dim=add_input_zero_dim,
        return_differences=predict_residual
    )
    val_data = dataset(
        bc_path=bc_path,
        swe_path=swe_path,
        set_type="valid",
        batch_size=batch_size,
        return_rotations=return_rotations,
        add_input_zero_dim=add_input_zero_dim,
        return_differences=predict_residual
    )

    #####################
    # Show model summary
    #####################
    model.summary()

    ###################
    # Define callbacks
    ###################
    term = tf.keras.callbacks.TerminateOnNaN()
    cp_callback_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=cp_callback_loss_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=True
    )
    cp_callback_vrmse = tf.keras.callbacks.ModelCheckpoint(
        filepath=cp_callback_metric_path,
        monitor="val_vrmse",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=True
    )
    callbacks = [term, cp_callback_loss, cp_callback_vrmse]
    if tensorboard_cb:
        tb_cb = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_callback_path,
            histogram_freq=1,
            write_graph=False,
            write_steps_per_second=True,
            update_freq="epoch",
            profile_batch=(1, 300)
        )
        callbacks.append(tb_cb)

    ##############
    # Train model
    ##############
    train_history = model.fit(
        x=train_data, batch_size=1, epochs=2, validation_data=val_data, callbacks=callbacks
    )
    with open(training_logs_json, "w") as f:
        json.dump(train_history.history, f, indent=4)

    ##############
    # Test models
    ##############
    for callback_path, rollout_statistics_path in [
        (cp_callback_loss_path, rollout_statistics_loss_path), (cp_callback_metric_path, rollout_statistics_metric_path)
    ]:
        loaded = tf.keras.models.load_model(
            callback_path,
            custom_objects={
                "ConvDirac": ConvDirac,
                "ConvGeodesic": ConvGeodesic,
                "AngularMaxPooling": AngularMaxPooling,
                "ConvHarmonic": ConvHarmonic,
                "BetaRelu": BetaRelu,
                "ConvGEM": ConvGEM,
                "ConvEMAN": ConvEMAN,
                "ConvGEMP": ConvGEMP,
                "ConvEMANP": ConvEMANP,
            }
        )
        test_data = dataset(
            bc_path=bc_path,
            swe_path=swe_path,
            set_type="test",
            batch_size=1,
            return_rotations=return_rotations,
            add_input_zero_dim=add_input_zero_dim,
            return_time_steps=False,
            return_differences=predict_residual
        )
        test_history = loaded.evaluate(test_data, verbose=1, return_dict=True)
        with open(rollout_statistics_path, "w") as f:
            json.dump(test_history, f, indent=4)
