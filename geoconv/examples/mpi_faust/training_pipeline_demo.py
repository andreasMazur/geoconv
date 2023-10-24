from geoconv.examples.mpi_faust.hypermodels.original.geodesic_hm import GeodesicHyperModel
from geoconv.examples.mpi_faust.hypermodels.lite.geodesic_lite_hm import GeodesicLiteHyperModel
from geoconv.examples.mpi_faust.hypermodels.original.dirac_hm import DiracHyperModel
from geoconv.examples.mpi_faust.hypermodels.lite.dirac_lite_hm import DiracLiteHyperModel
from geoconv.examples.mpi_faust.hypermodels.original.original_hm import OriginalHyperModel
from geoconv.examples.mpi_faust.hypermodels.lite.original_lite_hm import OriginalLiteHyperModel
from geoconv.examples.mpi_faust.hypermodels.original.geodesic_res_hm import GeoResHyperModel
from geoconv.examples.mpi_faust.hypermodels.lite.geodesic_res_lite_hm import GeoResLiteHyperModel
from geoconv.examples.mpi_faust.faust_data_set import load_preprocessed_faust
from geoconv.examples.mpi_faust.preprocess_faust import preprocess_faust
from geoconv.utils.measures import princeton_benchmark

from pathlib import Path

import keras_tuner
import tensorflow as tf


def training_pipeline_demo(reference_mesh_path,
                           signal_dim,
                           preprocessed_data,
                           n_radial=5,
                           n_angular=8,
                           registration_path="",
                           compute_shot=True,
                           geodesic_diameters_path="",
                           precomputed_gpc_radius=0.037,
                           save_gpc_systems=True,
                           template_radius=0.028,
                           logging_dir="./imcnn_training_logs",
                           output_dim=128,
                           splits=10,
                           amt_convolutions=1,
                           rotation_delta=1,
                           imcnn_variant="geodesic",
                           tuner_variant="hyperband"):
    """Executes preprocessing, hyperparameter-search and training on MPI-FAUST data set.

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
    splits: int
        [OPTIONAL] The amount of splits over which the ISC-layer should iterate
    amt_convolutions: int
        [IMCNN CONFIGURATION] The amount of geodesic convolutional layers for the IMCNN.
    rotation_delta: int
        [IMCNN CONFIGURATION] The rotation delta for the IMCNN.
    imcnn_variant: str
        [IMCNN CONFIGURATION] A string from ["dirac", "geodesic", "original", "geores"], which tells which type of IMCNN
        shall be used. You can train the corresponding lite-version by adding a "_lite" to the name passed for
        'imcnn_variant'. E.g. "dirac_lite".
    tuner_variant: str
        [HYPERPARAMETER TUNING CONFIGURATION] A string from ["hyperband", "bayesian"], which tells which hyperparameter
         optimization technique shall be used.
    """

    ######################
    # Preprocessing part
    ######################
    preprocess_zip = f"{preprocessed_data}.zip"
    if not Path(preprocess_zip).is_file():
        template_radius = preprocess_faust(
            n_radial=n_radial,
            n_angular=n_angular,
            target_dir=preprocessed_data,
            registration_path=registration_path,
            shot=compute_shot,
            geodesic_diameters_path=geodesic_diameters_path,
            precomputed_gpc_radius=precomputed_gpc_radius,
            save_gpc_systems=save_gpc_systems
        )
    else:
        print(f"Found preprocess-results: '{preprocess_zip}'. Skipping preprocessing.")

    if template_radius <= .0:
        raise RuntimeError("Please select a valid kernel radius >0.")

    ######################################
    # Hyperparameter tuning and training
    ######################################
    SIGNAL_DIM = signal_dim
    KERNEL_SIZE = (n_radial, n_angular)
    N_TARGET_NODES = 6890

    # Load data
    train_data = load_preprocessed_faust(preprocess_zip, signal_dim=SIGNAL_DIM, kernel_size=KERNEL_SIZE, set_type=0)
    val_data = load_preprocessed_faust(preprocess_zip, signal_dim=SIGNAL_DIM, kernel_size=KERNEL_SIZE, set_type=1)

    # Define hypermodel
    if imcnn_variant == "geodesic":
        imcnn = GeodesicHyperModel(
            SIGNAL_DIM,
            KERNEL_SIZE,
            N_TARGET_NODES,
            amt_convolutions,
            splits,
            splits,
            template_radius,
            rotation_delta,
            output_dim=output_dim
        )
    elif imcnn_variant == "geodesic_lite":
        imcnn = GeodesicLiteHyperModel(
            SIGNAL_DIM,
            KERNEL_SIZE,
            N_TARGET_NODES,
            amt_convolutions,
            splits,
            splits,
            template_radius,
            output_dim=output_dim
        )
    elif imcnn_variant == "dirac":
        imcnn = DiracHyperModel(
            SIGNAL_DIM,
            KERNEL_SIZE,
            N_TARGET_NODES,
            amt_convolutions,
            splits,
            splits,
            template_radius,
            rotation_delta,
            output_dim=output_dim
        )
    elif imcnn_variant == "dirac_lite":
        imcnn = DiracLiteHyperModel(
            SIGNAL_DIM,
            KERNEL_SIZE,
            N_TARGET_NODES,
            amt_convolutions,
            splits,
            splits,
            template_radius,
            output_dim=output_dim
        )
    elif imcnn_variant == "original":
        imcnn = OriginalHyperModel(
            SIGNAL_DIM, KERNEL_SIZE, splits, splits, template_radius, rotation_delta
        )
    elif imcnn_variant == "original_lite":
        imcnn = OriginalLiteHyperModel(
            SIGNAL_DIM, KERNEL_SIZE, splits, splits, template_radius
        )
    elif imcnn_variant == "geores":
        imcnn = GeoResHyperModel(
            SIGNAL_DIM,
            KERNEL_SIZE,
            N_TARGET_NODES,
            amt_convolutions,
            splits,
            splits,
            template_radius,
            rotation_delta,
            output_dim=output_dim
        )
    elif imcnn_variant == "geores_lite":
        imcnn = GeoResLiteHyperModel(
            SIGNAL_DIM,
            KERNEL_SIZE,
            N_TARGET_NODES,
            amt_convolutions,
            splits,
            splits,
            template_radius,
            output_dim=output_dim
        )
    else:
        raise RuntimeError("Choose a valid 'imcnn_variant'!")

    if tuner_variant == "hyperband":
        tuner = keras_tuner.Hyperband(
            hypermodel=imcnn,
            objective="val_loss",
            max_epochs=200,
            directory=f"{logging_dir}/keras_tuner",
            project_name=f"faust_example",
            seed=42
        )
    elif tuner_variant == "bayesian":
        tuner = keras_tuner.BayesianOptimization(
            hypermodel=imcnn,
            objective="val_loss",
            max_trials=1000,
            directory=f"{logging_dir}/keras_tuner",
            project_name=f"faust_example",
            seed=42
        )
    else:
        raise RuntimeError("Choose a valid 'tuner_variant'!")

    # Define callbacks
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=f"{logging_dir}/tensorboard",
        histogram_freq=1,
        write_graph=False,
        write_steps_per_second=True,
        update_freq="epoch",
        profile_batch=(1, 70 * 1)
    )

    # Start hyperparameter-search
    tuner.search(
        x=train_data.prefetch(tf.data.AUTOTUNE),
        validation_data=val_data.prefetch(tf.data.AUTOTUNE),
        callbacks=[stop, tb]
    )
    print(tuner.results_summary())

    # Save best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.build(input_shape=[(SIGNAL_DIM,), (KERNEL_SIZE[0], KERNEL_SIZE[1], 3, 2)])
    print(best_model.summary())
    best_model.save(f"{logging_dir}/best_model")

    # Evaluate best model with Princeton benchmark
    test_dataset = load_preprocessed_faust(preprocess_zip, signal_dim=SIGNAL_DIM, kernel_size=KERNEL_SIZE, set_type=2)
    princeton_benchmark(
        imcnn=best_model,
        test_dataset=test_dataset,
        ref_mesh_path=reference_mesh_path,
        file_name=f"{logging_dir}/best_model_benchmark"
    )
