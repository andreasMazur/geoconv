from tensorflow.keras.layers import Input
from gcnn_vertex_wise import ConvGeodesic

import tensorflow as tf


if __name__ == "__main__":

    signal = tf.random.uniform((10, 2, 4, 3, 5))
    barycentric_coords = tf.fill((10, 2, 4, 3), 0.33)

    # Define model
    signal_input = Input(shape=signal.shape[1:], name="signal")
    bary_input = Input(shape=barycentric_coords.shape[1:], name="Barycentric c.")
    geodesic_conv = ConvGeodesic(
        kernel_size=(2, 4), output_dim=32, amt_kernel=16, activation="relu"
    )([signal_input, bary_input])
    model = tf.keras.Model(inputs=[signal_input, bary_input], outputs=[geodesic_conv])

    output = model([signal, barycentric_coords])
    print(f"Output shape: {output.shape}")
