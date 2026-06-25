from geoconv.tensorflow.layers.activations.activation_beta_relu import BetaRelu
from geoconv.tensorflow.layers.descriptor.eucl_neighbors_descriptor import EuclNeighborsDescriptor
from geoconv.tensorflow.layers.experimental.lift_features import LiftFeatures2D
from geoconv.tensorflow.layers.experimental.unlift_features import UnLiftFeatures2D
from geoconv_examples.modelnet.architectures.deep_sets import DeepSet
from geoconv_examples.modelnet.training_configs.dictionaries import NORM_FACTORS_EUCL_DESCR_MN10
from geoconv.tensorflow.layers.convolutions.conv_eman_p import ConvEMANP

import tensorflow as tf


def define_hypermodel(hp,
                      input_types,
                      output_types,
                      template_radius,
                      n_radial,
                      n_angular):
    model = define_model(
        input_types=input_types,
        output_types=output_types,
        template_radius=template_radius,
        n_radial=n_radial,
        n_angular=n_angular,
        learning_rate=hp.Float("learning_rate", min_value=1e-8, max_value=0.01),
        lr_decay_rate=hp.Float("learning_rate_decay", min_value=0.5, max_value=0.999999)
    )
    return model


def define_model(input_types,
                 output_types,
                 template_radius,
                 n_radial,
                 n_angular,
                 learning_rate=0.001,
                 lr_decay_rate=1.0):
    """Builds and compiles an EMAN+-model for the ModelNet10 benchmark.

    Parameters
    ----------
    input_types: list
        A list of lists, where each list-element contains the input types for one EMAN+ layer.
    output_types: list
        A list of lists, where each list-element contains the output types for one EMAN+ layer.
    template_radius: float
        The template radius for the EMAN+ layers.
    n_radial: int
        The amount of radial coordinates used by the discretized template.
    n_angular: int
        The amount of angular coordinates used by the discretized template.
    learning_rate: float
        The learning rate for the EMAN+ model.
    lr_decay_rate: float
        The learning rate decay rate for the EMAN+ model.

    Returns
    -------
    tf.keras.Model:
        The EMAN+-model for the ModelNet10 benchmark.
    """
    # Define input layers
    vertices_input = tf.keras.Input(shape=(None, 3), name="vertices_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(None, n_radial, n_angular, 3, 3), name="bc_input", dtype=tf.float32)
    mask_input = tf.keras.Input(shape=(None,), name="mask_input", dtype=tf.bool)

    # Forward pass
    signal = EuclNeighborsDescriptor(n_neighbors=int(1 + n_radial * n_angular))(vertices_input)
    signal = tf.keras.layers.Normalization(
        axis=-1,
        mean=NORM_FACTORS_EUCL_DESCR_MN10[(n_radial, n_angular)]["mean"],
        variance=NORM_FACTORS_EUCL_DESCR_MN10[(n_radial, n_angular)]["variance"]
    )(signal)
    signal = LiftFeatures2D()(signal)
    for (it, ot) in zip(input_types, output_types):
        signal = ConvEMANP(
            template_radius=template_radius,
            input_types=it,
            output_types=ot,
            attention_types=ot
        )([signal, bc_input])
        signal = BetaRelu()(signal)

    # Aggregation and classification
    signal = UnLiftFeatures2D()(signal)
    output = DeepSet(
        local_network_dims=[],
        global_network_dims=[10],
        local_activation="linear",
        global_activation="linear"
    )([signal, mask_input])

    imcnn = tf.keras.Model(inputs=[vertices_input, bc_input, mask_input], outputs=output, name="mn10_model")
    imcnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=399_100,
                decay_rate=lr_decay_rate
            )
        ),
        metrics=["sparse_categorical_accuracy"]
    )
    imcnn.summary()
    return imcnn
