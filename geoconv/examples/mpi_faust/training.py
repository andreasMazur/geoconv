from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Normalization, Dropout, Reshape

from geoconv.examples.mpi_faust.model import PointCorrespondenceGeoCNN
from geoconv.layers.angular_max_pooling import AngularMaxPooling
from geoconv.layers.geodesic_conv import ConvGeodesic

import tensorflow as tf


def define_model(signal_shape,
                 bc_shape,
                 output_dim,
                 dataset_mean,
                 dataset_var,
                 amt_kernel=1,
                 rotation_delta=1,
                 lr=.00045,
                 dropout=.2):

    # Define model
    signal_input = Input(shape=signal_shape, name="signal")
    bary_input = Input(shape=bc_shape, name="barycentric")
    amp = AngularMaxPooling()

    signal = Normalization(axis=None, mean=dataset_mean, variance=dataset_var)(signal_input)
    signal = Dropout(rate=dropout)(signal)

    signal = Dense(16, activation="relu")(signal)

    signal = ConvGeodesic(
        output_dim=32,
        amt_kernel=amt_kernel,
        activation="relu",
        rotation_delta=rotation_delta
    )([signal, bary_input])
    signal = amp(signal)

    signal = ConvGeodesic(
        output_dim=64,
        amt_kernel=1,
        activation="relu",
        rotation_delta=rotation_delta
    )([signal, bary_input])
    signal = amp(signal)

    signal = ConvGeodesic(
        output_dim=128,
        amt_kernel=1,
        activation="relu",
        rotation_delta=rotation_delta
    )([signal, bary_input])
    signal = amp(signal)

    # signal = Dense(256, kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(signal)
    logits = Dense(output_dim, kernel_regularizer=tf.keras.regularizers.L2(l2=0.02))(signal)

    model = PointCorrespondenceGeoCNN(inputs=[signal_input, bary_input], outputs=[logits])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"], run_eagerly=True)
    model.summary()
    return model


def train_on_faust(tf_faust_dataset,
                   tf_faust_dataset_val,
                   model=None,
                   run=0):

    log_dir = f"./logs/fit/{run}/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="epoch", write_steps_per_second=True, profile_batch=(1, 1000)
    )

    checkpoint_path = f"./training/{run}_cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_freq='epoch', verbose=1)

    model.fit(
        tf_faust_dataset.prefetch(tf.data.AUTOTUNE),
        epochs=200,
        callbacks=[tensorboard_callback, cp_callback],
        validation_data=tf_faust_dataset_val.prefetch(tf.data.AUTOTUNE)
    )
