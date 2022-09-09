from tensorflow.keras import layers

from layers.angular_max_pooling import AngularMaxPooling
from layers.examples.mpi_faust.tf_dataset import load_preprocessed_faust, faust_mean_variance
from layers.geodesic_conv import ConvGeodesic

import keras_tuner as kt
import tensorflow as tf


class GeoConvHyperModel(kt.HyperModel):

    def __init__(self,
                 dataset_mean,
                 dataset_var,
                 input_dim=144,
                 amt_nodes=6890,
                 amt_target_nodes=6890,
                 kernel_size=(2, 4)):
        super().__init__()
        self.dataset_mean = dataset_mean
        self.dataset_var = dataset_var
        self.amt_nodes = amt_nodes
        self.amt_target_nodes = amt_target_nodes
        self.kernel_size = kernel_size
        self.input_dim = input_dim

    def build(self, hp):
        # Define model
        amp = AngularMaxPooling()
        signal_input = layers.Input(shape=(self.amt_nodes, self.input_dim), name="Signal")
        bary_input = layers.Input(
            shape=(self.amt_nodes, self.kernel_size[1], self.kernel_size[0], 3, 2), name="Barycentric"
        )
        signal = layers.Normalization(axis=None, mean=self.dataset_mean, variance=self.dataset_var)(signal_input)
        signal = layers.Dropout(rate=hp.Float(name="dropout_rate", min_value=.0, max_value=.3, step=.1))(signal)

        signal = layers.Dense(16, activation="relu")(signal)
        signal = ConvGeodesic(
            output_dim=hp.Int(name="output_dim_1", min_value=16, max_value=64, step=1),
            amt_kernel=1,
            activation="relu"
        )([signal, bary_input])
        signal = amp(signal)

        signal = ConvGeodesic(
            output_dim=hp.Int(name="output_dim_2", min_value=16, max_value=64, step=1),
            amt_kernel=1,
            activation="relu"
        )([signal, bary_input])
        signal = amp(signal)

        signal = ConvGeodesic(
            output_dim=hp.Int(name="output_dim_3", min_value=16, max_value=64, step=1),
            amt_kernel=1,
            activation="relu"
        )([signal, bary_input])
        signal = amp(signal)

        logits = layers.Dense(self.amt_target_nodes)(signal)

        # Declare model
        model = tf.keras.Model(inputs=[signal_input, bary_input], outputs=[logits])

        # Compile model
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.Adam(
            learning_rate=hp.Float(name="learning_rate", min_value=0.0001, max_value=0.1, step=0.0002)
        )
        model.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])

        return model


def faust_hypertuning(path_preprocessed_dataset,
                      run_id,
                      signal_dim=144,
                      amt_nodes=6890,
                      amt_target_nodes=6890,
                      kernel_size=(2, 4),
                      batch_size=1):

    # Load dataset
    tf_faust_dataset = load_preprocessed_faust(
        path_preprocessed_dataset, amt_nodes, signal_dim, kernel_size=kernel_size
    )
    tf_faust_dataset_val = load_preprocessed_faust(
        path_preprocessed_dataset, amt_nodes, signal_dim, kernel_size=kernel_size, val=True
    )
    faust_mean, faust_var = faust_mean_variance(tf_faust_dataset)

    # Declare hyperband-tuner
    tuner = kt.Hyperband(
        # Create hypermodel
        hypermodel=GeoConvHyperModel(
            dataset_mean=faust_mean,
            dataset_var=faust_var,
            input_dim=signal_dim,
            amt_nodes=amt_nodes,
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
        x=tf_faust_dataset.batch(batch_size).shuffle(5, reshuffle_each_iteration=True),
        validation_data=tf_faust_dataset_val.batch(batch_size).prefetch(tf.data.AUTOTUNE),
        epochs=300,
        callbacks=[stop]  # tensorboard
    )

    # Print best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:")
    for key, value in best_hp.values.items():
        print(key, value)
