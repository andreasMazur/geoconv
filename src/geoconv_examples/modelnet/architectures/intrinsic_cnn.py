from geoconv.tensorflow.layers import AngularMaxPooling, ConvDirac
from geoconv.tensorflow.layers import ConvGeodesic
from geoconv.tensorflow.layers.descriptor.eucl_neighbors_descriptor import EuclNeighborsDescriptor
from geoconv_examples.modelnet.architectures.deep_sets import DeepSet
from geoconv_examples.modelnet.dataset import MAX_N_VERTICES
from geoconv_examples.modelnet.training_configs.dictionaries import NORM_FACTORS_EUCL_DESCR_MN10

import tensorflow as tf


def define_hypermodel(hp,
                      output_dims,
                      template_radius,
                      n_radial,
                      n_angular,
                      kernel):
    tf.keras.backend.clear_session()
    model = define_model(
        output_dims=output_dims,
        template_radius=template_radius,
        n_radial=n_radial,
        n_angular=n_angular,
        kernel=kernel,
        learning_rate=hp.Float("learning_rate", min_value=1e-8, max_value=0.1),
        lr_decay_rate=hp.Float("learning_rate_decay", min_value=0.5, max_value=0.999999)
    )
    return model


def define_model(output_dims,
                 template_radius,
                 n_radial,
                 n_angular,
                 kernel,
                 learning_rate=0.001,
                 lr_decay_rate=1.0):
    # Define input layers
    vertices_input = tf.keras.Input(shape=(MAX_N_VERTICES, 3), name="vertices_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(MAX_N_VERTICES, n_radial, n_angular, 3, 2), name="bc_input", dtype=tf.float32)
    mask_input = tf.keras.Input(shape=(MAX_N_VERTICES,), name="mask_input", dtype=tf.bool)

    if kernel == "geodesic":
        layer_type = ConvGeodesic
    elif kernel == "dirac":
        layer_type = ConvDirac
    else:
        raise ValueError("The 'kernel' must be either 'geodesic' or 'dirac'.")

    # Forward pass
    signal = EuclNeighborsDescriptor(n_neighbors=int(1 + n_radial * n_angular))(vertices_input)
    signal = tf.keras.layers.Normalization(
        axis=-1,
        mean=NORM_FACTORS_EUCL_DESCR_MN10[(n_radial, n_angular)]["mean"],
        variance=NORM_FACTORS_EUCL_DESCR_MN10[(n_radial, n_angular)]["variance"]
    )(signal)
    for od in output_dims:
        signal = layer_type(
            output_dim=od,
            template_radius=template_radius,
            activation="relu",
            rotation_delta=1
        )([signal, bc_input])
        signal = AngularMaxPooling()(signal)

    # Aggregation and classification
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
