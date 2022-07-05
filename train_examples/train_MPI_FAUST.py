from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

from dataset.MPI_FAUST.tf_dataset import load_preprocessed_faust
from gcnn import ConvGeodesic

import tensorflow as tf
import datetime


def define_model(signal_shape, bc_shape):
    """Similar architecture to the one used in Section 7.2 in [1].

    Just the input SHOT-vector differs: Here it has 1056 entries. In [1] it has 150.

    [1]:
    > Jonathan Masci, Davide Boscaini, Michael M. Bronstein, Pierre Vandergheynst

    > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://www.cv-foundation.org/
    openaccess/content_iccv_2015_workshops/w22/html/Masci_Geodesic_Convolutional_Neural_ICCV_2015_paper.html)

    """
    signal_input = Input(shape=signal_shape, name="signal")
    bary_input = Input(shape=bc_shape, name="Barycentric c.")

    # LIN16+ReLU
    signal = Dense(16, activation="relu")(signal_input)
    # GC32+AMP+ReLU
    signal = ConvGeodesic(kernel_size=(2, 4), output_dim=32, amt_kernel=1, activation="relu")([signal, bary_input])
    # GC64+AMP+ReLU
    # signal = ConvGeodesic(kernel_size=(2, 4), output_dim=64, amt_kernel=1, activation="relu")([signal, bary_input])
    # GC128+AMP+ReLU
    # signal = ConvGeodesic(kernel_size=(2, 4), output_dim=128, amt_kernel=1, activation="relu")([signal, bary_input])
    # LIN256
    # signal = Dense(256)(signal)
    # LIN6890
    logits = Dense(6890)(signal)

    model = tf.keras.Model(inputs=[signal_input, bary_input], outputs=[logits])
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss, metrics=["categorical_accuracy"])
    model.summary()
    return model


def train():
    tf_faust_dataset = load_preprocessed_faust("/home/andreas/Uni/Masterarbeit/MPI-FAUST/preprocessed_faust.zip")
    model = define_model(signal_shape=(6890, 1056), bc_shape=(6890, 8, 8))

    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_path = "./training/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )

    model.fit(tf_faust_dataset, epochs=1, callbacks=[tensorboard_callback, cp_callback])


if __name__ == "__main__":
    train()
