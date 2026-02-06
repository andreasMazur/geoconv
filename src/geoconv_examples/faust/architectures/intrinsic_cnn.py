from geoconv.tensorflow.layers import AngularMaxPooling, ConvDirac
from geoconv.tensorflow.layers import ConvGeodesic
from geoconv.tensorflow.layers.descriptor.eucl_neighbors_descriptor import EuclNeighborsDescriptor
from geoconv_examples.faust.dataset import adapt_generator

import tensorflow as tf


def define_hypermodel(hp,
                      output_dims,
                      preprocess_method,
                      gpc_radius,
                      template_radius,
                      n_radial,
                      n_angular,
                      kernel,
                      faust_path):
    model = define_model(
        output_dims=output_dims,
        preprocess_method=preprocess_method,
        gpc_radius=gpc_radius,
        template_radius=template_radius,
        n_radial=n_radial,
        n_angular=n_angular,
        kernel=kernel,
        faust_path=faust_path,
        learning_rate=hp.Float("learning_rate", min_value=1e-8, max_value=0.1),
    )
    return model


def define_model(output_dims,
                 preprocess_method,
                 gpc_radius,
                 template_radius,
                 n_radial,
                 n_angular,
                 kernel,
                 faust_path,
                 learning_rate=0.001):
    # Define input layers
    vertices_input = tf.keras.Input(shape=(6890, 3), name="vertices_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(6890, n_radial, n_angular, 3, 2), name="bc_input", dtype=tf.float32)

    if kernel == "geodesic":
        layer_type = ConvGeodesic
    elif kernel == "dirac":
        layer_type = ConvDirac
    else:
        raise ValueError("The 'kernel' must be either 'geodesic' or 'dirac'.")

    # Remember descriptor- and normalization layer for normalization layer adaption
    descr_layer = EuclNeighborsDescriptor(n_radial, n_angular)
    normalization_layer = tf.keras.layers.Normalization(axis=-1)

    # Forward pass
    signal = descr_layer(vertices_input)
    signal = normalization_layer(signal)
    for od in output_dims:
        signal = layer_type(
            output_dim=od,
            template_radius=template_radius,
            activation="relu",
            rotation_delta=1
        )([signal, bc_input])
        signal = AngularMaxPooling()(signal)
    output = tf.keras.layers.Dense(6890, activation="linear")(signal)

    imcnn = tf.keras.Model(inputs=[vertices_input, bc_input], outputs=output, name="faust_model")
    imcnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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
