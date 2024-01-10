from geoconv.examples.mpi_faust.faust_data_set import load_preprocessed_faust
from geoconv.examples.mpi_faust.model import Imcnn
from geoconv.examples.mpi_faust.preprocess_faust import preprocess_faust
from geoconv.utils.measures import princeton_benchmark

from pathlib import Path
from tensorflow import keras

import tensorflow as tf


def train_model(reference_mesh_path,
                signal_dim,
                preprocessed_data,
                n_radial=5,
                n_angular=8,
                registration_path="",
                compute_shot=True,
                geodesic_diameters_path="",
                precomputed_gpc_radius=0.037,
                template_radius=0.028,
                logging_dir="./imcnn_training_logs",
                splits=10,
                rotation_delta=1,
                processes=1):
    """Trains one singular IMCNN

    Parameters
    ----------
    reference_mesh_path: str
        The path to the reference mesh file.
    signal_dim: int
        The dimensionality of the mesh signal
    preprocessed_data: str
        The path to the pre-processed data. If you have not pre-processed your data so far and saved it under the given
        path, this script will execute pre-processing for you. For this to work, you need to pass the arguments which
        are annotated with '[REQUIRED FOR PRE-PROCESSING]'. If pre-processing is not required, you can ignore those
        arguments.
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
    template_radius: float
        [OPTIONAL] The template radius of the ISC-layer (the one used during preprocessing, defaults to radius for FAUST
        data set).
    logging_dir: str
        [OPTIONAL] The path to the folder where logs will be stored
    splits: int
        [OPTIONAL] The amount of splits over which the ISC-layer should iterate
    rotation_delta: int
        [OPTIONAL] The amount of skips between each rotation while computing the convolution
    processes: int
        [OPTIONAL] The amount of concurrent processes that compute GPC-systems
    """

    # Load data
    preprocess_zip = f"{preprocessed_data}.zip"
    if not Path(preprocess_zip).is_file():
        template_radius = preprocess_faust(
            n_radial=n_radial,
            n_angular=n_angular,
            target_dir=preprocess_zip[:-4],
            registration_path=registration_path,
            shot=compute_shot,
            geodesic_diameters_path=geodesic_diameters_path,
            precomputed_gpc_radius=precomputed_gpc_radius,
            processes=processes
        )
    else:
        print(f"Found preprocess-results: '{preprocess_zip}'. Skipping preprocessing.")

    # Load data
    kernel_size = (n_radial, n_angular)
    train_data = load_preprocessed_faust(preprocess_zip, signal_dim=signal_dim, kernel_size=kernel_size, set_type=0)
    val_data = load_preprocessed_faust(preprocess_zip, signal_dim=signal_dim, kernel_size=kernel_size, set_type=1)

    # Define and compile model
    imcnn = Imcnn(template_radius=template_radius, splits=splits, rotation_delta=rotation_delta)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = keras.optimizers.Adam(learning_rate=0.00076215)
    imcnn.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])

    # Build model
    imcnn([tf.random.uniform(shape=(6890, signal_dim)), tf.random.uniform(shape=(6890,) + kernel_size + (3, 2))])
    imcnn.summary()

    # Define callbacks
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    tb = keras.callbacks.TensorBoard(
        log_dir=f"{logging_dir}/tensorboard",
        histogram_freq=1,
        write_graph=False,
        write_steps_per_second=True,
        update_freq="epoch",
        profile_batch=(1, 70)
    )

    imcnn.fit(x=train_data, callbacks=[stop, tb], validation_data=val_data, epochs=200)
    imcnn.save(f"{logging_dir}/saved_imcnn")

    # Evaluate best model with Princeton benchmark
    # test_dataset = load_preprocessed_faust(preprocess_zip, signal_dim=signal_dim, kernel_size=kernel_size, set_type=2)
    # princeton_benchmark(
    #     imcnn=imcnn,
    #     test_dataset=test_dataset,
    #     ref_mesh_path=reference_mesh_path,
    #     file_name=f"{train_data}/best_model_benchmark"
    # )
