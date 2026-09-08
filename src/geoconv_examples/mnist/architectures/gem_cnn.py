from geoconv.tensorflow.layers.activations.beta_relu import BetaRelu
from geoconv.tensorflow.layers.pooling.global_complex_max_pooling import GlobalComplexPooling
from geoconv.tensorflow.layers.convolutions.conv_gem import ConvGEM

import tensorflow as tf


def define_hypermodel(hp, input_types, output_types, template_radius, n_radial, n_angular):
    """Builds a KerasTuner hypermodel for the architecture.
        
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
        The amount of radial coordinates used by the discretized template.
    n_angular: int
        The amount of angular coordinates used by the discretized template.

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
        learning_rate=hp.Float("learning_rate", min_value=1e-8, max_value=0.01),
    )
    model.summary()
    return model


def define_model(input_types, output_types, template_radius, n_radial, n_angular, learning_rate=0.001):
    """Builds and compiles an GEM-CNN-model for the MNIST benchmark.

    Parameters
    ----------
    input_types: list
        A list of lists, where each list-element contains the input types for one GEM-CNN layer.
    output_types: list
        A list of lists, where each list-element contains the output types for one GEM-CNN layer.
    template_radius: float
        The template radius for the GEM-CNN layers.
    n_radial: int
        The amount of radial coordinates used by the discretized template.
    n_angular: int
        The amount of angular coordinates used by the discretized template.
    learning_rate: float
        The learning rate for the GEM-CNN model.

    Returns
    -------
    tf.keras.Model:
        The GEM-CNN-model for the MNIST benchmark.
    """
    image_size = 28 * 28

    # Define input layers
    image_input = tf.keras.Input(shape=(image_size, 2), name="image_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(image_size, n_radial, n_angular, 3, 2), name="bc_input", dtype=tf.float32)
    rotations_input = tf.keras.Input(shape=(image_size, image_size), name="rotations_input", dtype=tf.float32)

    # Initialize variables for forward pass
    signal = image_input

    # Forward pass
    for (it, ot) in zip(input_types, output_types):
        signal = ConvGEM(
            template_radius=template_radius,
            input_types=it,
            output_types=ot
        )([signal, bc_input, rotations_input])
        signal = BetaRelu()(signal)
    signal = GlobalComplexPooling()(signal)
    output = tf.keras.layers.Dense(10, activation="linear")(signal)

    imcnn = tf.keras.Model(
        inputs=[image_input, bc_input, rotations_input], outputs=output, name="mnist_model"
    )
    imcnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"]
    )
    return imcnn

