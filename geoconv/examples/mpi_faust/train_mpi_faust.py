from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Normalization, Dropout

from geoconv.examples.mpi_faust.tf_dataset import load_preprocessed_faust
from geoconv.geodesic_conv import ConvGeodesic

import tensorflow as tf
import datetime


def define_model(signal_shape, bc_shape, output_dim, layer_properties, lr=.00045):

    signal_input = Input(shape=signal_shape, name="Signal")
    bary_input = Input(shape=bc_shape, name="Barycentric")

    signal = Normalization()(signal_input)
    for (n_dim, n_kernel, dropout_rate) in layer_properties:
        signal = ConvGeodesic(
            kernel_size=(bc_shape[2], bc_shape[1]), output_dim=n_dim, amt_kernel=n_kernel, activation="relu"
        )([signal, bary_input])
        if dropout_rate:
            signal = Dropout(rate=dropout_rate)(signal)
    logits = Dense(output_dim)(signal)

    model = tf.keras.Model(inputs=[signal_input, bary_input], outputs=[logits])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])
    model.summary()
    return model


def train_on_faust(path_to_preprocessing_result, lr=.00045, batch_size=1, amt_nodes=6890, model=None):
    tf_faust_dataset = load_preprocessed_faust(path_to_preprocessing_result, amt_nodes)
    tf_faust_dataset_val = load_preprocessed_faust(path_to_preprocessing_result, amt_nodes, val=True)

    if model is None:
        model = define_model(
            signal_shape=(amt_nodes, 1056),
            bc_shape=(amt_nodes, 4, 2, 6),
            lr=lr,
            output_dim=amt_nodes,
            layer_properties=[(256, 2, 0.2), (256, 2, 0.)]
        )

    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="epoch", write_steps_per_second=True, profile_batch=(1, 1000)
    )

    checkpoint_path = "./training/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_freq='epoch', verbose=1)

    model.fit(
        tf_faust_dataset.batch(batch_size).shuffle(5, reshuffle_each_iteration=True),
        epochs=10,
        callbacks=[tensorboard_callback, cp_callback],
        validation_data=tf_faust_dataset_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
