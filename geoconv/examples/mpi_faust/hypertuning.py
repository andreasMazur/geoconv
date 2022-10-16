from tensorflow.keras import layers

from geoconv.examples.mpi_faust.model import PointCorrespondenceGeoCNN
from geoconv.layers.angular_max_pooling import AngularMaxPooling
from geoconv.examples.mpi_faust.tf_dataset import load_preprocessed_faust, faust_mean_variance
from geoconv.layers.geodesic_conv import ConvGeodesic

import keras_tuner as kt
import tensorflow as tf


class GeoConvHyperModel(kt.HyperModel):

    def __init__(self,
                 dataset_mean,
                 dataset_var,
                 input_dim=3,
                 amt_target_nodes=6890,
                 kernel_size=(3, 8)):
        super().__init__()
        self.dataset_mean = dataset_mean
        self.dataset_var = dataset_var
        self.amt_target_nodes = amt_target_nodes
        self.kernel_size = kernel_size
        self.input_dim = input_dim

    def build(self, hp):
        # Define model
        amp = AngularMaxPooling()
        signal_input = layers.Input(shape=(self.input_dim,), name="Signal")
        bary_input = layers.Input(
            shape=(self.kernel_size[0], self.kernel_size[1], 3, 2), name="Barycentric"
        )
        signal = layers.Normalization(axis=None, mean=self.dataset_mean, variance=self.dataset_var)(signal_input)
        signal = layers.Dropout(rate=hp.Float(name="dropout_rate", min_value=.0, max_value=.3, step=.1))(signal)

        signal = layers.Dense(16, activation="relu")(signal)

        # amt_kernel = hp.Int("amt_kernel", min_value=1, max_value=4, step=1,)
        signal = ConvGeodesic(
            output_dim=32, amt_kernel=4, activation="relu", rotation_delta=2, splits=26
        )([signal, bary_input])
        signal = amp(signal)

        signal = ConvGeodesic(
            output_dim=64, amt_kernel=4, activation="relu", rotation_delta=2, splits=26
        )([signal, bary_input])
        signal = amp(signal)

        signal = ConvGeodesic(
            output_dim=128, amt_kernel=4, activation="relu", rotation_delta=2, splits=26
        )([signal, bary_input])
        signal = amp(signal)

        signal = layers.Dense(256)(signal)
        logits = layers.Dense(self.amt_target_nodes)(signal)

        # Declare model
        model = PointCorrespondenceGeoCNN(inputs=[signal_input, bary_input], outputs=[logits])

        # Compile model
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )
        opt = tf.keras.optimizers.Adam(
            learning_rate=hp.Float(name="learning_rate", min_value=0.00001, max_value=0.001, step=0.00001),
            beta_1=hp.Float(name="beta_1", min_value=0.8, max_value=0.99, step=0.01),
            beta_2=hp.Float(name="beta_2", min_value=0.998, max_value=0.9999, step=0.0001),
        )
        model.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])

        return model


def faust_hypertuning(path_preprocessed_dataset,
                      run_id,
                      signal_dim=3,
                      amt_target_nodes=6890,
                      kernel_size=(3, 8)):

    # Load dataset
    tf_faust_dataset = load_preprocessed_faust(
        path_preprocessed_dataset, signal_dim, kernel_size=kernel_size
    )
    tf_faust_dataset_val = load_preprocessed_faust(
        path_preprocessed_dataset, signal_dim, kernel_size=kernel_size, val=True
    )
    faust_mean, faust_var = faust_mean_variance(tf_faust_dataset)

    # Declare hyperband-tuner
    tuner = kt.Hyperband(
        # Create hypermodel
        hypermodel=GeoConvHyperModel(
            dataset_mean=faust_mean,
            dataset_var=faust_var,
            input_dim=signal_dim,
            amt_target_nodes=amt_target_nodes,
            kernel_size=kernel_size
        ),
        objective="val_sparse_categorical_accuracy",
        max_epochs=200,
        factor=3,
        directory=f"./logs/{run_id}/",
        project_name=f"faust_{run_id}"
    )
    # Create callbacks
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # Tensorboard callback requires too much space
    # tensorboard = tf.keras.callbacks.TensorBoard(
    #     log_dir=f"./logs/fit/{run_id}/",
    #     histogram_freq=1,
    #     update_freq="epoch",
    #     write_steps_per_second=True,
    #     profile_batch=(1, 1000)
    # )

    # Initiate hyperparameter search
    tuner.search(
        x=tf_faust_dataset.prefetch(tf.data.AUTOTUNE),
        validation_data=tf_faust_dataset_val.prefetch(tf.data.AUTOTUNE),
        epochs=200,
        callbacks=[stop]  # tensorboard
    )

    # Print best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:")
    for key, value in best_hp.values.items():
        print(key, value)
