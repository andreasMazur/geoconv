from tensorflow.keras.layers import Input
from gcnn import ConvGeodesic

import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    signal = np.load("/home/andreas/Uni/Masterarbeit/MPI-FAUST/training/registrations_SHOT/tr_reg_000.npy")
    barycentric_coords = np.load("./misc/test_bary_coords.npy", allow_pickle=True)

    # Define model
    signal_input = Input(shape=signal.shape, name="signal")
    bary_input = Input(shape=barycentric_coords.shape, name="Barycentric c.")
    geodesic_conv = ConvGeodesic(
        kernel_size=(2, 4),
        output_dim=1,
        amt_kernel=5,
        activation="relu"
    )([signal_input, bary_input])
    model = tf.keras.Model(inputs=[signal_input, bary_input], outputs=[geodesic_conv])

    signal = tf.expand_dims(signal, axis=0)
    barycentric_coords = tf.expand_dims(barycentric_coords, axis=0)
    output = model([signal, barycentric_coords])
    print(f"Output shape: {output.shape}")
