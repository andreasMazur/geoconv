from geoconv.tensorflow.layers import AngularMaxPooling, ConvDirac
from geoconv.tensorflow.layers import ConvGeodesic

import tensorflow as tf


def define_hypermodel(hp, output_dims, template_radius, n_radial, n_angular, kernel):
    """Builds a KerasTuner hypermodel for the architecture.

    Parameters
    ----------
    hp: kt.HyperParameters
        The hyperparameter object.
    output_dims: list
        The output dims.
    template_radius: float
        The template radius.
    n_radial: int
        The amount of radial coordinates used by the discretized template.
    n_angular: int
        The amount of angular coordinates used by the discretized template.
    kernel: str
        The weighting function used to interpolated template vertex features among each other..

    Returns
    -------
    tf.keras.Model
        The constructed model.
    """
    model = define_model(
        output_dims=output_dims,
        template_radius=template_radius,
        n_radial=n_radial,
        n_angular=n_angular,
        kernel=kernel,
        learning_rate=hp.Float("learning_rate", min_value=1e-8, max_value=0.01),
    )
    model.summary()
    return model


def define_model(output_dims, template_radius, n_radial, n_angular, kernel, learning_rate=0.001):
    """Builds and compiles a ISC/GCNN-model for the MNIST benchmark.

    Parameters
    ----------
    output_dims: list
        A list of integer, where each element describes the output dimensions for one ISC/GCNN layer.
    template_radius: float
        The template radius for the ISC/GCNN layers.
    n_radial: int
        The amount of radial coordinates used by the discretized template.
    n_angular: int
        The amount of angular coordinates used by the discretized template.
    kernel: str
        Either 'geodesic' to build GCNNs or 'dirac' to build ISCs.
    learning_rate: float
        The learning rate for the ISC/GCNN model.

    Returns
    -------
    tf.keras.Model:
        The ISC/GCNN-model for the MNIST benchmark.
    """
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
    signal = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")(signal)
    output = tf.keras.layers.Dense(10, activation="linear")(signal)

    imcnn = tf.keras.Model(inputs=[image_input, bc_input], outputs=output, name="mnist_model")
    imcnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"]
    )
    return imcnn
