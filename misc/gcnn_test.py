from tensorflow.keras.layers import Input

from gcnn import ConvGeodesic
from dataset.MPI_FAUST.tf_dataset import load_preprocessed_faust

import tensorflow as tf


if __name__ == "__main__":
    gen = load_preprocessed_faust(
        "/home/andreas/PycharmProjects/Masterarbeit/dataset/MPI_FAUST/preprocessed_registrations.zip"
    ).as_numpy_iterator()
    (signal, barycentric_coords), _ = next(gen)

    # Define model
    signal_input = Input(shape=signal.shape[1:], name="signal")
    bary_input = Input(shape=barycentric_coords.shape[1:], name="Barycentric c.")
    geodesic_conv = ConvGeodesic(
        kernel_size=(2, 4), output_dim=1, amt_kernel=5, activation="relu"
    )([signal_input, bary_input])
    model = tf.keras.Model(inputs=[signal_input, bary_input], outputs=[geodesic_conv])

    print(f"Input shape: {signal.shape}")
    output = model([signal, barycentric_coords])
    print(f"Output shape: {output.shape}")
