from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Normalization, Dropout

from geoconv.examples.mpi_faust.tf_dataset import load_preprocessed_faust
from geoconv.geodesic_conv import ConvGeodesic
from geoconv.ResNetBlock import ResNetBlock

import tensorflow as tf


def define_model_paper(signal_shape, bc_shape, kernel_amt):
    signal_input = Input(shape=signal_shape, name="Signal")
    bary_input = Input(shape=bc_shape, name="Barycentric")

    signal = Dense(16, activation="relu")(signal_input)
    signal = ConvGeodesic(
        kernel_size=(bc_shape[2], bc_shape[1]), output_dim=32, amt_kernel=kernel_amt, activation="relu"
    )([signal, bary_input])
    signal = ConvGeodesic(
        kernel_size=(bc_shape[2], bc_shape[1]), output_dim=64, amt_kernel=kernel_amt, activation="relu"
    )([signal, bary_input])
    signal = ConvGeodesic(
        kernel_size=(bc_shape[2], bc_shape[1]), output_dim=128, amt_kernel=kernel_amt, activation="relu"
    )([signal, bary_input])
    signal = Dense(256)(signal)
    signal_output = Dense(6890)(signal)

    model = tf.keras.Model(inputs=[signal_input, bary_input], outputs=[signal_output])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])
    model.summary()
    return model


def define_res_model(signal_shape, bc_shape, output_dim, layer_properties, lr=.00045, dropout=.2):

    signal_input = Input(shape=signal_shape, name="Signal")
    bary_input = Input(shape=bc_shape, name="Barycentric")

    signal = Normalization()(signal_input)
    signal = Dropout(rate=dropout)(signal)
    for (n_dim, n_kernel, dropout_rate) in layer_properties:
        signal = ResNetBlock(
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


def define_model(signal_shape, bc_shape, output_dim, layer_properties, lr=.00045, dropout=.2):

    signal_input = Input(shape=signal_shape, name="Signal")
    bary_input = Input(shape=bc_shape, name="Barycentric")

    signal = Normalization()(signal_input)
    signal = Dropout(rate=dropout)(signal)
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


def train_on_faust(path_to_preprocessing_result,
                   lr=.00045,
                   batch_size=1,
                   amt_nodes=6890,
                   signal_dim=1056,
                   model=None,
                   run=0):
    tf_faust_dataset = load_preprocessed_faust(path_to_preprocessing_result, amt_nodes, signal_dim)
    tf_faust_dataset_val = load_preprocessed_faust(path_to_preprocessing_result, amt_nodes, signal_dim, val=True)

    if model is None:
        model = define_model(
            signal_shape=(amt_nodes, signal_dim),
            bc_shape=(amt_nodes, 4, 2, 6),
            lr=lr,
            output_dim=amt_nodes,
            layer_properties=[(256, 2, 0.2), (256, 2, 0.)]
        )

    log_dir = f"./logs/fit/{run}/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="epoch", write_steps_per_second=True, profile_batch=(1, 1000)
    )

    checkpoint_path = f"./training/{run}_cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_freq='epoch', verbose=1)

    model.fit(
        tf_faust_dataset.batch(batch_size).shuffle(5, reshuffle_each_iteration=True),
        epochs=200,
        callbacks=[tensorboard_callback, cp_callback],
        validation_data=tf_faust_dataset_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
