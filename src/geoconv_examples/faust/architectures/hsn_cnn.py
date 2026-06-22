from geoconv.tensorflow.layers.activation_beta_relu import BetaRelu
from geoconv.tensorflow.layers.convolutions.conv_harmonic import ConvHarmonic
from geoconv.tensorflow.layers.descriptor.eucl_neighbors_descriptor import EuclNeighborsDescriptor
from geoconv_examples.faust.dataset import adapt_generator

import tensorflow as tf


def define_hypermodel(hp, output_dims, preprocess_method, gpc_radius, template_radius, n_radial, n_angular, faust_path):
    model = define_model(
        output_dims=output_dims,
        preprocess_method=preprocess_method,
        gpc_radius=gpc_radius,
        template_radius=template_radius,
        n_radial=n_radial,
        n_angular=n_angular,
        faust_path=faust_path,
        learning_rate=hp.Float("learning_rate", min_value=1e-8, max_value=0.1),
        lr_decay_rate=hp.Float("learning_rate_decay", min_value=0.5, max_value=0.999999)
    )
    return model


def define_model(output_dims,
                 preprocess_method,
                 gpc_radius,
                 template_radius,
                 n_radial,
                 n_angular,
                 faust_path,
                 learning_rate=0.001,
                 lr_decay_rate=1.0):
    """Builds and compiles a HSN-model for the FAUST benchmark.

    Parameters
    ----------
    output_dims: list
        A list of integer, where each element describes the output dimensions for one HSN layer.
    preprocess_method: str
        The used charting algorithm.
    gpc_radius: float
        The used maximum radius for the charting algorithm.
    template_radius: float
        The template radius for the HSN layers.
    n_radial: int
        The amount of radial coordinates used by the discretized template.
    n_angular: int
        The amount of angular coordinates used by the discretized template.
    faust_path: str
        The path to the preprocessed FAUST dataset.
    learning_rate: float
        The learning rate for the HSN model.
    lr_decay_rate: float
        The learning rate decay rate for the HSN model.

    Returns
    -------
    tf.keras.Model:
        The HSN-model for the FAUST benchmark.
    """
    # Define input layers
    vertices_input = tf.keras.Input(shape=(6890, 3), name="vertices_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(6890, n_radial, n_angular, 3, 3), name="bc_input", dtype=tf.float32)

    # Remember descriptor- and normalization layer for normalization layer adaption
    descr_layer = EuclNeighborsDescriptor(n_neighbors=int(1 + n_radial * n_angular))
    normalization_layer = tf.keras.layers.Normalization(axis=-1)

    # Forward pass
    signal = descr_layer(vertices_input)
    signal = normalization_layer(signal)
    for idx, od in enumerate(output_dims):
        signal = ConvHarmonic(
            output_dim=od,
            template_radius=template_radius,
            activation="linear",
            rotation_order=idx+1
        )([signal, bc_input])
        signal = BetaRelu()(signal)
    output = tf.keras.layers.Dense(6890, activation="linear")(signal)

    imcnn = tf.keras.Model(
        inputs=[vertices_input, bc_input], outputs=output, name="faust_model"
    )
    imcnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=7000,
                decay_rate=lr_decay_rate
            )
        ),
        metrics=["sparse_categorical_accuracy"]
    )

    # Adapt normalization
    normalization_layer.adapt(
        adapt_generator(
            faust_path,
            "train",
            n_radial,
            n_angular,
            preprocess_method,
            gpc_radius,
            template_radius,
            descr_layer
        )
    )
    return imcnn
