from geoconv.tensorflow.layers import ConvDirac, ConvGeodesic, AngularMaxPooling
from geoconv.tensorflow.layers.activation_beta_relu import BetaRelu
from geoconv.tensorflow.layers.convolutions.conv_harmonic import ConvHarmonic
from geoconv.tensorflow.layers.lift_features import LiftFeatures2D
from geoconv_gem.tensorflow.layers.convolutions.conv_gem import ConvGEM
from geoconv_gem.tensorflow.layers.convolutions.conv_eman import ConvEMAN
from geoconv_gem.tensorflow.layers.convolutions.conv_gem_p import ConvGEMP
from geoconv_gem.tensorflow.layers.convolutions.conv_eman_p import ConvEMANP
from geoconv_examples.planetswe.dataset import dataset

import os
import tensorflow as tf
import numpy as np
import random
import json


def training(model,
             bc_path,
             swe_path,
             return_rotations,
             save_path,
             epochs=10,
             random_seed=42,
             tensorboard_cb=False,
             batch_size=1,
             add_input_zero_dim=False):
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
    cp_callback_vrmse = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path_acc,
        monitor="val_vrmse",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=True
    )
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=0.001)
    callbacks = [term, cp_callback_loss, cp_callback_vrmse, stop]

    if tensorboard_cb:
        tb_cb = tf.keras.callbacks.TensorBoard(
            log_dir=f"{os.path.dirname(save_path)}/tensorboard",
            histogram_freq=1,
            write_graph=False,
            write_steps_per_second=True,
            update_freq="epoch",
            profile_batch=(1, 300)
        )
        callbacks.append(tb_cb)

    train_history = model.fit(
        x=train_data, batch_size=1, epochs=epochs, validation_data=val_data, callbacks=callbacks
    )

    # Test best performing model
    model = tf.keras.models.load_model(
        save_path,
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
        batch_size=batch_size,
        return_rotations=return_rotations,
        add_input_zero_dim=add_input_zero_dim
    )
    test_history = model.evaluate(test_data, return_dict=True)

    # Save history
    with open(f"{save_path[:-6]}_train_history.json", "w") as f:
        json.dump(train_history.history, f, indent=4)
    with open(test_saving_path, "w") as f:
        json.dump(test_history, f, indent=4)
