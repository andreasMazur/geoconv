from geoconv.tensorflow.layers.conv_harmonic import ConvHarmonic
from geoconv.tensorflow.layers.pooling.global_complex_max_pooling import GlobalComplexPooling
from geoconv.tensorflow.layers.activation_beta_relu import BetaRelu
from geoconv_examples.mnist.dataset import dataset

import tensorflow as tf
import keras_tuner as kt
import json


def define_model(output_dims, template_radius, n_radial, n_angular, activation, learning_rate=0.001):
    image_size = 28 * 28

    # Define input layers
    image_input = tf.keras.Input(shape=(image_size, 2), name="image_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(image_size, n_radial, n_angular, 3, 2), name="bc_input", dtype=tf.float32)
    rotations_input = tf.keras.Input(shape=(image_size, image_size), name="rotations_input", dtype=tf.float32)

    # Initialize variables for forward pass
    signal = image_input

    # Forward pass
    for od in output_dims:
        signal = ConvHarmonic(
            output_dim=od,
            template_radius=template_radius,
            activation=activation,
            rotation_order=1
        )([signal, bc_input, rotations_input])
        signal = BetaRelu()(signal)
    signal = GlobalComplexPooling()(signal)
    output = tf.keras.layers.Dense(10, activation="linear")(signal)

    imcnn = tf.keras.Model(
        inputs=[image_input, bc_input, rotations_input], outputs=output, name="mnist_model"
    )
    imcnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"]
    )
    return imcnn


def hypertuning(mnist_atlas, n_radial, n_angular, batch_size, activation, save_path, epochs=5):
    # Get data
    train_data, template_radius = dataset(
        mnist_atlas, set_type="train", n_radial=n_radial, n_angular=n_angular, batch_size=batch_size
    )
    test_data, _ = dataset(
        mnist_atlas, set_type="test", n_radial=n_radial, n_angular=n_angular, batch_size=batch_size
    )

    # Define hypermodel function
    def get_hypermodel(hp):
        model = define_model(
            output_dims=[
                hp.Int("units", min_value=8, max_value=64, step=2) for _ in range(hp.Int("n_layers", min_value=4, max_value=16))
            ],
            template_radius=template_radius,
            n_radial=n_radial,
            n_angular=n_angular,
            activation=activation,
            learning_rate=hp.Float("learning_rate", min_value=1e-6, max_value=0.1),
        )
        model.summary()
        return model

    # Hyperparameter search
    tuner = kt.BayesianOptimization(
        hypermodel=get_hypermodel,
        objective=kt.Objective("val_sparse_categorical_accuracy", direction="max"),
        max_trials=10_000,
        num_initial_points=10,
        seed=42
    )
    tuner.search(train_data, epochs=epochs, validation_data=test_data)

    # Save best model
    best_model = tuner.get_best_models()[0]
    if save_path[-6:] != ".keras":
        save_path += ".keras"
    best_model.save(save_path)


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
