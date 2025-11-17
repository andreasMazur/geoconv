from geoconv.tensorflow.layers.conv_gauge_equiv import ConvGaugeEquiv
from geoconv_examples.mnist.dataset import dataset

import tensorflow as tf
import json


def define_model(output_dims, template_radius, n_radial, n_angular, activation):
    image_size = 28 * 28

    # Define input layers
    image_input = tf.keras.Input(shape=(image_size, 2), name="image_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(image_size, n_radial, n_angular, 3, 2), name="bc_input", dtype=tf.float32)
    rotations_input = tf.keras.Input(shape=(image_size, image_size), name="rotations_input", dtype=tf.float32)
    rotation_orders_input = tf.keras.Input(shape=(1,), name="orders_input", dtype=tf.float32)

    # Initialize variables for forward pass
    signal = image_input
    rotation_orders = rotation_orders_input

    # Forward pass
    for od in output_dims:
        signal, rotation_orders = ConvGaugeEquiv(
            output_dim=od,
            template_radius=template_radius,
            activation=activation
        )([signal, bc_input, rotations_input, rotation_orders])
    signal = tf.keras.layers.GlobalMaxPool1D()(signal)
    output = tf.keras.layers.Dense(10, activation="linear")(signal)

    imcnn = tf.keras.Model(
        inputs=[image_input, bc_input, rotations_input, rotation_orders_input], outputs=output, name="mnist_model"
    )
    imcnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["sparse_categorical_accuracy"]
    )
    return imcnn


def training(mnist_atlas, n_radial, n_angular, batch_size, output_dims, save_path, activation, epochs=10):
    # Get data
    train_data, template_radius = dataset(
        mnist_atlas, set_type="train", n_radial=n_radial, n_angular=n_angular, batch_size=batch_size
    )
    test_data, _ = dataset(
        mnist_atlas, set_type="test", n_radial=n_radial, n_angular=n_angular, batch_size=batch_size
    )

    # Get model
    model = define_model(
        output_dims=output_dims,
        template_radius=template_radius,
        n_radial=n_radial,
        n_angular=n_angular,
        activation=activation
    )
    model.summary()

    # Train model
    history = model.fit(x=train_data, batch_size=batch_size, epochs=epochs, validation_data=test_data)

    # Save model
    if save_path[-6:] != ".keras":
        save_path += ".keras"
    model.save(save_path)

    # Save history
    with open(f"{save_path[:-6]}_history.json", "w") as f:
        json.dump(history.history, f, indent=4)
