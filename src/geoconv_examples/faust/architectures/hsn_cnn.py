from geoconv.tensorflow.layers import PointCloudShotDescriptor
from geoconv.tensorflow.layers.activation_beta_relu import BetaRelu
from geoconv.tensorflow.layers.convolutions.conv_harmonic import ConvHarmonic
from geoconv_examples.faust.dataset import adapt_generator

import tensorflow as tf


def define_hypermodel(hp, output_dims, template_radius, n_radial, n_angular, faust_path):
    model = define_model(
        output_dims=output_dims,
        template_radius=template_radius,
        n_radial=n_radial,
        n_angular=n_angular,
        learning_rate=hp.Float("learning_rate", min_value=1e-8, max_value=0.01),
    )
    # Adapt normalization layer
    normalization_layer = [l for l in model.layers if "normalization" == l.name][0]
    descr_layer = [l for l in model.layers if l.name == "point_cloud_shot_descriptor"][0]
    normalization_layer.adapt(
        adapt_generator(faust_path, "train", n_radial, n_angular, template_radius, descr_layer)
    )
    model.summary()
    return model


def define_model(output_dims, template_radius, n_radial, n_angular, learning_rate=0.001):
    # Define input layers
    vertices_input = tf.keras.Input(shape=(6890, 3), name="vertices_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(6890, n_radial, n_angular, 3, 2), name="bc_input", dtype=tf.float32)
    rotations_input = tf.keras.Input(shape=(6890, 6890), name="rotations_input", dtype=tf.float32)

    # Forward pass
    signal = PointCloudShotDescriptor(n_radial, n_angular)(vertices_input)
    signal = tf.keras.layers.Normalization(axis=-1)(signal)
    signal = tf.keras.layers.Dense(64, activation="relu")(signal)
    signal = tf.keras.layers.LayerNormalization(axis=-1)(signal)
    for od in output_dims:
        signal = ConvHarmonic(
            output_dim=od,
            template_radius=template_radius,
            activation="linear",
            rotation_order=1
        )([signal, bc_input, rotations_input])
        signal = BetaRelu()(signal)
    output = tf.keras.layers.Dense(6890, activation="linear")(signal)

    imcnn = tf.keras.Model(
        inputs=[vertices_input, bc_input, rotations_input], outputs=output, name="faust_model"
    )
    imcnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"]
    )
    return imcnn
