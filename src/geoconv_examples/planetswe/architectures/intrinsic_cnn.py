from geoconv.tensorflow.layers import AngularMaxPooling, ConvDirac
from geoconv.tensorflow.layers import ConvGeodesic
from geoconv_examples.planetswe.vrmse import VRMSE

import tensorflow as tf


def define_model(n_radial, n_angular, template_radius, kernel, output_dims, learning_rate, lr_decay_rate):
    # Define input layers
    features_input = tf.keras.Input(shape=(131_072, 3), name="features_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(131_072, n_radial, n_angular, 3, 2), name="bc_input", dtype=tf.float32)

    if kernel == "geodesic":
        layer_type = ConvGeodesic
    elif kernel == "dirac":
        layer_type = ConvDirac
    else:
        raise ValueError("The 'kernel' must be either 'geodesic' or 'dirac'.")

    # Forward pass
    signal = features_input
    for od in output_dims:
        signal = layer_type(
            output_dim=od,
            template_radius=template_radius,
            activation="relu",
            rotation_delta=1
        )([signal, bc_input])
        signal = AngularMaxPooling()(signal)

    # Predict height
    height_prediction = tf.keras.layers.Dense(1, activation="linear")(signal)

    # Predict velocity
    velocity_prediction = layer_type(
        output_dim=2,
        template_radius=template_radius,
        activation="linear",
        rotation_delta=1
    )([signal, bc_input])
    velocity_prediction = AngularMaxPooling()(velocity_prediction)

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
        metrics=[VRMSE()]
    )
    return model
