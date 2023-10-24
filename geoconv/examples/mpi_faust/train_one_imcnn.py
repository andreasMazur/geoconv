from geoconv.examples.mpi_faust.faust_data_set import load_preprocessed_faust
from geoconv.examples.mpi_faust.preprocess_faust import preprocess_faust
from geoconv.layers.lite.conv_dirac_lite import ConvDiracLite
from geoconv.models.intrinsic_model import ImCNN
from geoconv.utils.measures import princeton_benchmark

from pathlib import Path
from tensorflow import keras

import tensorflow as tf


def define_model(input_dim,
                 n_radial,
                 n_angular,
                 output_dim=128,
                 template_radius=0.028,
                 amt_templates=1,
                 splits=10):
    """Define a shallow IMCNN

    Parameters
    ----------
    input_dim: int
        The dimensionality of the input for the ISC-layer.
    n_radial: int
        The amount of radial coordinates for the template.
    n_angular: int
        The amount of angular coordinates for the template.
    output_dim: int
        The dimensionality of the output of the ISC-layer
    template_radius: float
        The template radius of the ISC-layer (the one used during preprocessing, defaults to radius for FAUST data set)
    amt_templates: int
        The amount of templates for the ISC-layer
    splits: int
        The amount of splits over which the ISC-layer should iterate
    """

    signal_input = keras.layers.Input(shape=(input_dim,), name="signal")
    bc_input = keras.layers.Input(shape=(n_radial, n_angular, 3, 2), name="bc")

    signal = ConvDiracLite(
        output_dim=output_dim,
        amt_templates=amt_templates,
        template_radius=template_radius,
        activation="relu",
        name="ISC_layer",
        splits=splits,
    )([signal_input, bc_input])
    output = keras.layers.Dense(6890)(signal)

    model = ImCNN(splits=splits, inputs=[signal_input, bc_input], outputs=[output])
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = keras.optimizers.Adam(learning_rate=0.0073429)
    model.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])

    return model


def train_model(reference_mesh_path,
                signal_dim,
                preprocessed_data,
                n_radial=5,
                n_angular=8,
                registration_path="",
                compute_shot=True,
                geodesic_diameters_path="",
                precomputed_gpc_radius=0.037,
                save_gpc_systems=False,
                template_radius=0.028,
                logging_dir="./imcnn_training_logs",
                output_dim=128,
                amt_templates=1,
                splits=10):
    """Trains one singular IMCNN

    Parameters
    ----------
    reference_mesh_path: str
        The path to the reference mesh file.
    signal_dim: int
        The dimensionality of the mesh signal
    preprocessed_data: str
        The path to the pre-processed data. If you have not pre-processed your data so far and saved it under the given
        path, this script will execute pre-processing for you.
        pre-processing for you.
    n_radial: int
        [REQUIRED FOR PRE-PROCESSING] The amount of radial coordinates for the template.
    n_angular: int
        [REQUIRED FOR PRE-PROCESSING] The amount of angular coordinates for the template.
    registration_path: str
        [REQUIRED FOR PRE-PROCESSING] The path of the training-registration files in the FAUST data set.
    compute_shot: bool
        [REQUIRED FOR PRE-PROCESSING] Whether to compute SHOT-descriptors during preprocessing as the mesh signal
    geodesic_diameters_path: str
        [REQUIRED FOR PRE-PROCESSING] The path to pre-computed geodesic diameters for the FAUST-registration meshes.
    precomputed_gpc_radius: float
        [REQUIRED FOR PRE-PROCESSING] The GPC-system radius to use for GPC-system computation. If not provided, the
        script will calculate it.
    save_gpc_systems: bool
        [REQUIRED FOR PRE-PROCESSING] Whether to save the GPC-systems.
    template_radius: float
        [OPTIONAL] The template radius of the ISC-layer (the one used during preprocessing, defaults to radius for FAUST
        data set).
    logging_dir: str
        [OPTIONAL] The path to the folder where logs will be stored
    output_dim: int
        [OPTIONAL] The dimensionality of the output of the ISC-layer
    amt_templates: int
        [OPTIONAL] The amount of templates for the ISC-layer
    splits: int
        [OPTIONAL] The amount of splits over which the ISC-layer should iterate
    """

    # Load data
    preprocess_zip = f"{preprocessed_data}.zip"
    if not Path(preprocess_zip).is_file():
        template_radius = preprocess_faust(
            n_radial=n_radial,
            n_angular=n_angular,
            target_dir=preprocess_zip,
            registration_path=registration_path,
            shot=compute_shot,
            geodesic_diameters_path=geodesic_diameters_path,
            precomputed_gpc_radius=precomputed_gpc_radius,
            save_gpc_systems=save_gpc_systems
        )
    else:
        print(f"Found preprocess-results: '{preprocess_zip}'. Skipping preprocessing.")

    kernel_size = (n_radial, n_angular)
    train_data = load_preprocessed_faust(preprocess_zip, signal_dim=signal_dim, kernel_size=kernel_size, set_type=0)
    val_data = load_preprocessed_faust(preprocess_zip, signal_dim=signal_dim, kernel_size=kernel_size, set_type=1)

    # Model
    imcnn = define_model(signal_dim, n_radial, n_angular, output_dim, template_radius, amt_templates, splits)
    imcnn.summary()

    # Define callbacks
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=f"{logging_dir}/tensorboard",
        histogram_freq=1,
        write_graph=False,
        write_steps_per_second=True,
        update_freq="epoch",
        profile_batch=(1, 70)
    )

    imcnn.fit(x=train_data, callbacks=[stop, tb], validation_data=val_data, epochs=3)
    imcnn.save(f"{logging_dir}/saved_imcnn")

    # Evaluate best model with Princeton benchmark
    test_dataset = load_preprocessed_faust(preprocess_zip, signal_dim=signal_dim, kernel_size=kernel_size, set_type=2)
    princeton_benchmark(
        imcnn=imcnn,
        test_dataset=test_dataset,
        ref_mesh_path=reference_mesh_path,
        file_name=f"{train_data}/best_model_benchmark"
    )
