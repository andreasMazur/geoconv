from geoconv.tensorflow.layers import ConvDirac, ConvGeodesic, AngularMaxPooling
from geoconv.tensorflow.layers.activation_beta_relu import BetaRelu
from geoconv.tensorflow.layers.convolutions.conv_harmonic import ConvHarmonic
from geoconv_gem.tensorflow.layers.convolutions.conv_gem import ConvGEM
from geoconv_gem.tensorflow.layers.convolutions.conv_eman import ConvEMAN
from geoconv_gem.tensorflow.layers.convolutions.conv_gem_p import ConvGEMP
from geoconv_gem.tensorflow.layers.convolutions.conv_eman_p import ConvEMANP
from geoconv_examples.planetswe.dataset import dataset
from geoconv_examples.planetswe.training_configs.dictionaries import PLANETSWE_NORM_VALUES
from geoconv_examples.planetswe.vrmse import compute_vrmse

import os
import tensorflow as tf
import numpy as np
import random
import json


def rollout_benchmark(model, test_data, t_max, save_path, zero_pad=False):
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
        time_step_errors.append(vrmse_t_next)
        print(f"\rt: {time_step % t_max} -> t+1 {(time_step % t_max) + 1}: VRMSE(t+1) = {vrmse_t_next}")

        # Zero pad prediction if required for architecture
        if zero_pad:
            prediction = tf.concat([prediction, tf.zeros(tf.shape(prediction)[:2])[..., None]], axis=-1)
    time_step_errors = np.array(time_step_errors).reshape(-1, t_max)
    np.save(save_path, time_step_errors)


def training(model,
             bc_path,
             swe_path,
             return_rotations,
             save_path,
             epochs=10,
             random_seed=42,
             tensorboard_cb=False,
             batch_size=1,
             add_input_zero_dim=False,
             rollout_t_max=100):
    # Define saving paths
    os.makedirs(save_path, exist_ok=True)
    tensorboard_callback_path = f"{save_path}/tensorboard"
    cp_callback_loss_path = f"{save_path}/loss_callback.keras"
    cp_callback_metric_path = f"{save_path}/metric_callback.keras"
    rollout_statistics_loss_path = f"{save_path}/loss_rollout_statistics.npy"
    rollout_statistics_metric_path = f"{save_path}/metric_rollout_statistics.npy"
    training_logs_json = f"{save_path}/training_logs.json"

    # Check if metric rollout already has been computed - last benchmark in training function
    if os.path.isfile(rollout_statistics_metric_path):
        print(f"{rollout_statistics_metric_path} already exists! Skipping training...")
        return

    # Set seeds
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Get data
    train_data = dataset(
        bc_path=bc_path,
        swe_path=swe_path,
        set_type="train",
        batch_size=batch_size,
        return_rotations=return_rotations,
        add_input_zero_dim=add_input_zero_dim
    )
    val_data = dataset(
        bc_path=bc_path,
        swe_path=swe_path,
        set_type="valid",
        batch_size=batch_size,
        return_rotations=return_rotations,
        add_input_zero_dim=add_input_zero_dim
    )

    # Show model summary
    model.summary()

    # Train model
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
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=True
    )
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=2, min_delta=0.001)
    callbacks = [term, cp_callback_loss, cp_callback_vrmse, stop]

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

    # Model training
    train_history = model.fit(
        x=train_data, batch_size=1, epochs=epochs, validation_data=val_data, callbacks=callbacks
    )
    with open(training_logs_json, "w") as f:
        json.dump(train_history.history, f, indent=4)

    # Test best performing model
    for callback_path, rollout_statistics_path in [
        (cp_callback_loss_path, rollout_statistics_loss_path), (cp_callback_metric_path, rollout_statistics_metric_path)
    ]:
        model = tf.keras.models.load_model(
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

        # Compute rollout statistics
        test_data = dataset(
            bc_path=bc_path,
            swe_path=swe_path,
            set_type="test",
            batch_size=1,
            return_rotations=return_rotations,
            add_input_zero_dim=add_input_zero_dim,
            max_time_steps=rollout_t_max,
            return_time_steps=True
        )
        rollout_benchmark(
            model,
            test_data,
            t_max=rollout_t_max,
            save_path=rollout_statistics_path,
            zero_pad=add_input_zero_dim
        )
