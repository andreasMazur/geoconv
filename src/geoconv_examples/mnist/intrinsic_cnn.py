from geoconv.tensorflow.layers import AngularMaxPooling, ConvDirac
from geoconv.tensorflow.layers import ConvGeodesic
from geoconv.tensorflow.layers.activation_beta_relu import BetaRelu

import tensorflow as tf


def define_model(output_dims, template_radius, n_radial, n_angular, kernel, learning_rate=0.001):
    image_size = 28 * 28

    # Define input layers
    image_input = tf.keras.Input(shape=(image_size, 2), name="image_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(image_size, n_radial, n_angular, 3, 2), name="bc_input", dtype=tf.float32)

    # Initialize variables for forward pass
    signal = image_input

    if kernel == "geodesic":
        layer_type = ConvGeodesic
    elif kernel == "dirac":
        layer_type = ConvDirac
    else:
        raise ValueError("The 'kernel' must be either 'geodesic' or 'dirac'.")

    # Forward pass
    for od in output_dims:
        signal = layer_type(
            output_dim=od,
            template_radius=template_radius,
            activation="relu",
            rotation_delta=1
        )([signal, bc_input])
        signal = AngularMaxPooling()(signal)
        signal = BetaRelu()(signal)
    signal = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")(signal)
    output = tf.keras.layers.Dense(10, activation="linear")(signal)

    imcnn = tf.keras.Model(inputs=[image_input, bc_input], outputs=output, name="mnist_model")
    imcnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"]
    )
    return imcnn
