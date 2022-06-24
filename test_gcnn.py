from gcnn import ConvGeodesic

import numpy as np
import tensorflow as tf


if __name__ == "__main__":

    # Define model
    barycentric_coords = np.load("./misc/test_bary_coords.npy", allow_pickle=True)
    layer = ConvGeodesic(
        kernel_size=(2, 4),
        output_dim=1,
        barycentric_coordinates=barycentric_coords,
        activation="relu"
    )
    model = tf.keras.Sequential([layer])

    # Pass signal through model
    signal = np.load("/home/andreas/Uni/Masterarbeit/MPI-FAUST/training/registrations_SHOT/tr_reg_000.npy")
    output = model(signal)
