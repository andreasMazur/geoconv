from geoconv.tensorflow.layers.activations.beta_relu import BetaRelu
from geoconv.tensorflow.layers.descriptor.eucl_neighbors_descriptor import EuclNeighborsDescriptor
from geoconv.tensorflow.layers.experimental.lift_features import LiftFeatures2D
from geoconv_examples.faust.dataset import adapt_dataset

from geoconv.tensorflow.layers.convolutions.conv_eman import ConvEMAN

import tensorflow as tf


def define_hypermodel(hp,
                      input_types,
                      output_types,
                      preprocess_method,
                      gpc_radius,
                      template_radius,
                      n_radial,
                      n_angular,
                      faust_path):
    model = define_model(
        input_types=input_types,
        output_types=output_types,
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


def define_model(input_types,
                 output_types,
                 preprocess_method,
                 gpc_radius,
                 template_radius,
                 n_radial,
                 n_angular,
                 faust_path,
                 learning_rate=0.001,
                 lr_decay_rate=1.0):
    """Builds and compiles an EMAN-model for the FAUST benchmark.

    Parameters
    ----------
    input_types: list
        A list of lists, where each list-element contains the input types for one EMAN layer.
    output_types: list
        A list of lists, where each list-element contains the output types for one EMAN layer.
    preprocess_method: str
        The used charting algorithm.
    gpc_radius: float
        The used maximum radius for the charting algorithm.
    template_radius: float
        The template radius for the EMAN layers.
    n_radial: int
        The amount of radial coordinates used by the discretized template.
    n_angular: int
        The amount of angular coordinates used by the discretized template.
    faust_path: str
        The path to the preprocessed FAUST dataset.
    learning_rate: float
        The learning rate for the EMAN model.
    lr_decay_rate: float
        The learning rate decay rate for the EMAN model.

    Returns
    -------
    tf.keras.Model:
        The EMAN-model for the FAUST benchmark.
    """
    # Define input layers
    vertices_input = tf.keras.Input(shape=(6890, 3), name="vertices_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(6890, n_radial, n_angular, 3, 3), name="bc_input", dtype=tf.float32)

    # Remember descriptor- and normalization layer for normalization layer adaption
    descr_layer = EuclNeighborsDescriptor(n_neighbors=int(1 + n_radial * n_angular))
    normalization_layer = tf.keras.layers.Normalization(axis=-1)
    lift = LiftFeatures2D()

    # Forward pass
    signal = descr_layer(vertices_input)
    signal = normalization_layer(signal)
    signal = lift(signal)
    for (it, ot) in zip(input_types, output_types):
        signal = ConvEMAN(
            template_radius=template_radius,
            input_types=it,
            output_types=ot,
            attention_types=ot
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
    imcnn.summary()

    # Adapt normalization
    normalization_layer.adapt(
        adapt_dataset(
            faust_path,
            "train",
            n_radial,
            n_angular,
            preprocess_method,
            tf.constant(gpc_radius, dtype=tf.float64),
            tf.constant(template_radius, dtype=tf.float64),
            n_neighbors=int(1 + n_radial * n_angular)
        )
    )

    return imcnn
