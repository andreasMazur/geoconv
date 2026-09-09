from geoconv.tensorflow.layers.activations.beta_relu import BetaRelu
from geoconv_examples.planetswe.vrmse import VRMSE
from geoconv.tensorflow.layers.convolutions.conv_gem import ConvGEM

import tensorflow as tf


def define_hypermodel(hp,
                      input_types,
                      output_types,
                      template_radius,
                      n_radial,
                      n_angular,
                      predict_residual):
    """Builds a PlanetSWE GEM-CNN hypermodel for KerasTuner.

    Parameters
    ----------
    hp: kt.HyperParameters
        The hyperparameter object.
    input_types: list
        The input types.
    output_types: list
        The output types.
    template_radius: float
        The template radius.
    n_radial: int
        The number of radial coordinates of the template.
    n_angular: int
        The number of angular coordinates of the template.
    predict_residual: bool
        Whether the model should predict residuals which are added onto the current state instead of the full state for
        the next time step.

    Returns
    -------
    tf.keras.Model
        The constructed model.
    """
    model = define_model(
        input_types=input_types,
        output_types=output_types,
        template_radius=template_radius,
        n_radial=n_radial,
        n_angular=n_angular,
        learning_rate=hp.Float("learning_rate", min_value=0.0007, max_value=0.003),
        lr_decay_rate=hp.Float("learning_rate_decay", min_value=0.8, max_value=0.999999),
        predict_residual=predict_residual
    )
    return model


def define_model(input_types,
                 output_types,
                 template_radius,
                 n_radial,
                 n_angular,
                 learning_rate,
                 lr_decay_rate,
                 predict_residual):
    """Builds the PlanetSWE GEM-CNN model.

    Parameters
    ----------
    input_types: list
        The input types.
    output_types: list
        The output types.
    template_radius: float
        The template radius.
    n_radial: int
        The number of radial coordinates of the template.
    n_angular: int
        The number of angular coordinates of the template.
    learning_rate: float
        The learning rate.
    lr_decay_rate: float
        The lr decay rate.
    predict_residual: bool
        Whether the model should predict residuals which are added onto the current state instead of the full state for
        the next time step.

    Returns
    -------
    tf.keras.Model
        The constructed model.
    """
    # Define input layers
    features_input = tf.keras.Input(shape=(131_072, 4), name="features_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(131_072, n_radial, n_angular, 3, 3), name="bc_input", dtype=tf.float32)

    # Forward pass
    signal = features_input
    for (it, ot) in zip(input_types, output_types):
        signal = ConvGEM(
            template_radius=template_radius,
            input_types=it,
            output_types=ot
        )([signal, bc_input])
        signal = BetaRelu()(signal)

    # Predict height
    height_prediction = tf.keras.layers.Dense(1, activation="linear")(signal)

    # Predict velocity
    velocity_prediction = ConvGEM(
        template_radius=template_radius,
        input_types=output_types[-1],
        output_types=[1]
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
