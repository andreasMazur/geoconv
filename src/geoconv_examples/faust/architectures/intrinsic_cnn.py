from geoconv.tensorflow.layers import AngularMaxPooling, ConvDirac, PointCloudShotDescriptor
from geoconv.tensorflow.layers import ConvGeodesic
from geoconv.tensorflow.layers.descriptor.eucl_neighbors_descriptor import EuclNeighborsDescriptor

import tensorflow as tf


def define_hypermodel(hp, output_dims, template_radius, n_radial, n_angular, kernel):
    model = define_model(
        output_dims=output_dims,
        template_radius=template_radius,
        n_radial=n_radial,
        n_angular=n_angular,
        kernel=kernel,
        learning_rate=hp.Float("learning_rate", min_value=1e-8, max_value=0.1),
    )
    model.summary()
    return model


def define_model(output_dims, template_radius, n_radial, n_angular, kernel, learning_rate=0.001):
    # Define input layers
    vertices_input = tf.keras.Input(shape=(6890, 3), name="vertices_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(6890, n_radial, n_angular, 3, 2), name="bc_input", dtype=tf.float32)

    if kernel == "geodesic":
        layer_type = ConvGeodesic
    elif kernel == "dirac":
        layer_type = ConvDirac
    else:
        raise ValueError("The 'kernel' must be either 'geodesic' or 'dirac'.")

    # Forward pass
    signal = PointCloudShotDescriptor(n_radial, n_angular)(vertices_input)
    signal = tf.keras.layers.Normalization(axis=-1)(signal)
    signal = tf.keras.layers.Dense(64, activation="relu")(signal)
    signal = tf.keras.layers.LayerNormalization(axis=-1)(signal)
    for od in output_dims:
        signal = layer_type(
            output_dim=od,
            template_radius=template_radius,
            activation="relu",
            rotation_delta=1
        )([signal, bc_input])
        signal = tf.keras.layers.LayerNormalization(axis=-1)(signal)
        signal = AngularMaxPooling()(signal)
    output = tf.keras.layers.Dense(6890, activation="linear")(signal)

    imcnn = tf.keras.Model(inputs=[vertices_input, bc_input], outputs=output, name="faust_model")
    imcnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"]
    )
    return imcnn
