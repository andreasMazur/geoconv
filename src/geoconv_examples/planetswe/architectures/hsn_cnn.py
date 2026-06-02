from geoconv.tensorflow.layers.activation_beta_relu import BetaRelu
from geoconv.tensorflow.layers.convolutions.conv_harmonic import ConvHarmonic

import tensorflow as tf


def define_model(n_radial, n_angular, template_radius, output_dims, learning_rate, lr_decay_rate):
    # Define input layers
    features_input = tf.keras.Input(shape=(131_072, 3), name="features_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(131_072, n_radial, n_angular, 3, 3), name="bc_input", dtype=tf.float32)

    # Forward pass
    signal = features_input
    for idx, od in enumerate(output_dims):
        signal = ConvHarmonic(
            output_dim=od,
            template_radius=template_radius,
            activation="linear",
            rotation_order=idx + 1
        )([signal, bc_input])
        signal = BetaRelu()(signal)

    # Predict height
    height_prediction = tf.keras.layers.Dense(1, activation="linear")(signal)

    # Predict velocity
    velocity_prediction = ConvHarmonic(
        output_dim=2,
        template_radius=template_radius,
        activation="linear",
        rotation_order=idx + 1
    )([signal, bc_input])

    # Concatenate predictions
    output = tf.keras.layers.Concatenate(axis=-1)([height_prediction, velocity_prediction])

    # Compile model
    model = tf.keras.Model(inputs=[features_input, bc_input], outputs=output, name="planetswe_model")
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=7_000,
                decay_rate=lr_decay_rate
            )
        ),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return model
