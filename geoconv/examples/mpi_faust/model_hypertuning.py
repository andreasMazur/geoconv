from geoconv.examples.mpi_faust.faust_data_set import load_preprocessed_faust
from geoconv.layers.angular_max_pooling import AngularMaxPooling
from geoconv.layers.conv_dirac import ConvDirac
from tensorflow import keras

import keras_tuner
import tensorflow as tf
import os


class HyperModel(keras_tuner.HyperModel):

    def __init__(self, signal_dim, kernel_size, template_radius, splits, rotation_delta):
        super().__init__()
        self.signal_dim = signal_dim
        self.kernel_size = kernel_size
        self.template_radius = template_radius
        self.splits = splits
        self.rotation_delta = rotation_delta
        self.output_dims = [96, 256, 384, 384, 256]

    def build(self, hp):
        signal_input = keras.layers.Input(shape=self.signal_dim, name="Signal_input")
        bc_input = keras.layers.Input(shape=(self.kernel_size[0], self.kernel_size[1], 3, 2), name="BC_input")
        amp = AngularMaxPooling()

        signal = keras.layers.Dense(64, activation="relu", name="Downsize")(signal_input)
        signal = keras.layers.BatchNormalization(axis=-1, name="BN_downsize")(signal)
        for idx in range(len(self.output_dims)):
            signal = ConvDirac(
                amt_templates=self.output_dims[idx],
                template_radius=self.template_radius,
                activation="relu",
                name=f"ISC_layer_{idx}",
                splits=self.splits,
                rotation_delta=self.rotation_delta,
                template_regularizer=tf.keras.regularizers.L2(l2=hp.Float("lr", min_value=1e-5, max_value=1e-1)),
                bias_regularizer=tf.keras.regularizers.L2(l2=hp.Float("lr", min_value=1e-5, max_value=1e-1))
            )([signal, bc_input])
            signal = amp(signal)
            signal = keras.layers.BatchNormalization(axis=-1, name=f"BN_layer_{idx}")(signal)
        output = keras.layers.Dense(6890, name="Output")(signal)

        model = keras.Model(inputs=[signal_input, bc_input], outputs=[output])
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = keras.optimizers.Adam(learning_rate=hp.Float("lr", min_value=1e-5, max_value=1e-1))
        model.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])
        return model


def hypertune(logging_dir,
              preprocessed_data,
              signal_dim,
              n_radial,
              n_angular,
              template_radius,
              splits,
              rotation_delta):
    """Tunes the learning rate of the above IMCNN.

    Parameters
    ----------
    logging_dir: str
        The path to the logging directory. If nonexistent, directory will be created.
    preprocessed_data: str
        The path to the preprocessed data (the *.zip-file).
    signal_dim:
        The dimensionality of the signal.
    n_radial:
        The amount of radial coordinates of the kernel.
    n_angular:
        The amount of angular coordinates of the kernel.
    template_radius:
        The template radius.
    splits:
        The amount of splits for the ISC-layers.
    rotation_delta:
        The rotation delta for the ISC-layers.
    """
    # Create logging dir if necessary
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Load data
    preprocess_zip = f"{preprocessed_data}.zip"
    kernel_size = (n_radial, n_angular)
    train_data = load_preprocessed_faust(preprocess_zip, signal_dim=signal_dim, kernel_size=kernel_size, set_type=0)
    val_data = load_preprocessed_faust(preprocess_zip, signal_dim=signal_dim, kernel_size=kernel_size, set_type=1)

    # Load hypermodel
    hyper = HyperModel(
        signal_dim=signal_dim,
        kernel_size=kernel_size,
        template_radius=template_radius,
        splits=splits,
        rotation_delta=rotation_delta
    )

    # Configure tuner
    tuner = keras_tuner.Hyperband(
        hypermodel=hyper,
        objective="val_loss",
        max_epochs=200,
        directory=f"{logging_dir}/keras_tuner",
        project_name=f"faust_example",
        seed=42
    )

    # Start hyperparameter-search
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    tuner.search(
        x=train_data.prefetch(tf.data.AUTOTUNE),
        validation_data=val_data.prefetch(tf.data.AUTOTUNE),
        callbacks=[stop]
    )
    print(tuner.results_summary())

    # Save best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.build(input_shape=[(signal_dim,), (n_radial, n_angular, 3, 2)])
    print(best_model.summary())
    best_model.save(f"{logging_dir}/best_model")
