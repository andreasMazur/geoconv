from geoconv_examples.faust.dataset import dataset

import tensorflow as tf
import json


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
