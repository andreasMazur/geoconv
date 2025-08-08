from geoconv.tensorflow.layers import ConvDirac, ConvGeodesic, ConvZero, AngularMaxPooling
from geoconv_examples.faust.tensorflow.dataset import N_VERTICES, SIG_DIM

import tensorflow as tf


def build_faust_classifier(variant, n_radial, n_angular, isc_layer_dims, template_radius, rotation_delta):
    if variant is None or variant == "dirac":
        layer_type = ConvDirac
    elif variant == "geodesic":
        layer_type = ConvGeodesic
    elif variant == "zero":
        layer_type = ConvZero
    else:
        raise RuntimeError("Select a layer type from: ['dirac', 'geodesic', 'zero']")

    shape_features = tf.keras.Input(shape=(N_VERTICES, SIG_DIM), name="features_input", dtype=tf.float32)
    bc_input = tf.keras.Input(shape=(N_VERTICES, n_radial, n_angular, 3, 2), name="bc_input", dtype=tf.float32)
    signal = shape_features
    for n in isc_layer_dims:
        signal = layer_type(
            amt_templates=n,
            template_radius=template_radius,
            activation="elu",
            rotation_delta=rotation_delta
        )([signal, bc_input])
        signal = AngularMaxPooling()(signal)
    prediction = tf.keras.layers.Dense(N_VERTICES, activation="linear")(signal)
    return tf.keras.Model(inputs=[shape_features, bc_input], outputs=prediction)
