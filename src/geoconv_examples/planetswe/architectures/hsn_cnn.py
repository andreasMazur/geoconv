from geoconv.tensorflow.layers.activations.beta_relu import BetaRelu
from geoconv.tensorflow.layers.convolutions.conv_harmonic import ConvHarmonic
from geoconv_examples.planetswe.vrmse import VRMSE

import tensorflow as tf


def define_hypermodel(hp,
                      output_dims,
                      template_radius,
                      n_radial,
                      n_angular,
                      predict_residual):
    model = define_model(
        n_radial=n_radial,
        n_angular=n_angular,
        template_radius=template_radius,
        output_dims=output_dims,
        learning_rate=hp.Float("learning_rate", min_value=0.0007, max_value=0.003),
        lr_decay_rate=hp.Float("learning_rate_decay", min_value=0.9, max_value=0.999999),
        predict_residual=predict_residual
    )
    return model


def define_model(n_radial, n_angular, template_radius, output_dims, learning_rate, lr_decay_rate, predict_residual):
    # Define input layers
    features_input = tf.keras.Input(shape=(131_072, 4), name="features_input", dtype=tf.float32)
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
    output = tf.keras.layers.Concatenate(axis=-1)([velocity_prediction, height_prediction])
    if predict_residual:
        output = tf.keras.layers.Add()([features_input[..., :3], output])

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
        metrics=[VRMSE()]
    )
    return model
