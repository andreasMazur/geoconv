from geoconv_examples.mpi_faust.tensorflow.faust_data_set import faust_generator
from geoconv_examples.mpi_faust.tensorflow.faust_data_set import load_preprocessed_faust
from geoconv_examples.mpi_faust.tensorflow.model import Imcnn
from geoconv_examples.mpi_faust.data.preprocess_faust import preprocess_faust
from geoconv_examples.mpi_faust_segmentation.tensorflow.convert_dataset import imcnn_new_dataset

from pathlib import Path

import tensorflow as tf
import keras
import numpy as np
import sys


def run_training(signal_dim,
                 kernel_size,
                 template_radius,
                 layer_conf,
                 model,
                 segmentation,
                 init_lr,
                 weight_decay,
                 preprocess_zip,
                 logging_dir,
                 exp_number,
                 train_data,
                 epochs,
                 val_data=None):
    # Define and compile model
    imcnn = Imcnn(
        signal_dim=signal_dim,
        kernel_size=kernel_size,
        template_radius=template_radius,
        layer_conf=layer_conf,
        variant=model,
        segmentation=segmentation
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = keras.optimizers.AdamW(
        learning_rate=keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=init_lr,
            decay_steps=500,
            decay_rate=0.99
        ),
        weight_decay=weight_decay
    )
    imcnn.compile(optimizer=opt, loss=loss)

    # Adapt normalization
    print("Initializing normalization layer..")
    imcnn.normalize.build(tf.TensorShape([6890, signal_dim]))
    adaption_data = load_preprocessed_faust(
        preprocess_zip, signal_dim=signal_dim, kernel_size=kernel_size, set_type=0, only_signal=True
    )
    imcnn.normalize.adapt(adaption_data)
    print("Done.")

    # Build model
    imcnn([tf.random.uniform(shape=(6890, signal_dim)), tf.random.uniform(shape=(6890,) + kernel_size + (3, 2))])
    imcnn.summary()

    # Define callbacks
    csv = keras.callbacks.CSVLogger(f"{logging_dir}/training_{exp_number}.log")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
    tb = keras.callbacks.TensorBoard(
        log_dir=f"{logging_dir}/tensorboard_{exp_number}",
        histogram_freq=1,
        write_graph=False,
        write_steps_per_second=True,
        update_freq="epoch",
        profile_batch=(1, 70)
    )

    # Train and save model
    if val_data is None:
        imcnn.fit(x=train_data, callbacks=[tb, csv], epochs=epochs)
    else:
        imcnn.fit(x=train_data, callbacks=[stop, tb, csv], validation_data=val_data, epochs=epochs)
    imcnn.save(f"{logging_dir}/saved_imcnn_{exp_number}")

    return imcnn


def train_model(preprocessed_data,
                new_dataset_path,
                n_radial=5,
                n_angular=8,
                registration_path="",
                compute_shot=True,
                precomputed_gpc_radius=0.037,
                template_radius=0.027744965069279016,
                logging_dir="./imcnn_training_logs",
                processes=1,
                init_lr=0.00165,
                weight_decay=0.005,
                layer_conf=None,
                model_variant="dirac",
                add_noise=False,
                segmentation=False,
                save_coordinates=True,
                epochs=200,
                seeds=None):
    """Trains one singular IMCNN

    Parameters
    ----------
    preprocessed_data: str
        The path to the pre-processed data. If you have not pre-processed your data so far and saved it under the given
        path, this script will execute pre-processing for you. For this to work, you need to pass the arguments which
        are annotated with '[REQUIRED FOR PRE-PROCESSING]'. If pre-processing is not required, you can ignore those
        arguments.
    new_dataset_path: str
        TODO
    n_radial: int
        [REQUIRED FOR PRE-PROCESSING] The amount of radial coordinates for the template.
    n_angular: int
        [REQUIRED FOR PRE-PROCESSING] The amount of angular coordinates for the template.
    registration_path: str
        [REQUIRED FOR PRE-PROCESSING] The path of the training-registration files in the FAUST data set.
    compute_shot: bool
        [REQUIRED FOR PRE-PROCESSING] Whether to compute SHOT-descriptors during preprocessing as the mesh signal
    precomputed_gpc_radius: float
        [REQUIRED FOR PRE-PROCESSING] The GPC-system radius to use for GPC-system computation. If not provided, the
        script will calculate it.
    template_radius: float
        [OPTIONAL] The template radius of the ISC-layer (the one used during preprocessing, defaults to radius for FAUST
        data set).
    logging_dir: str
        [OPTIONAL] The path to the folder where logs will be stored
    processes: int
        [OPTIONAL] The amount of concurrent processes. Affects preprocessing and Princeton benchmark.
    init_lr: float
        [OPTIONAL] Initial learning rate.
    weight_decay: float
        [OPTIONAL] Weight decay.
    layer_conf: list
        [OPTIONAL] List of tuples: The first entry references the output dimensions of the i-th ISC-layer, The second
        entry references of skips between each rotation while computing the convolution (rotation delta).
    model_variant: str
        [OPTIONAL] Which model variant (['dirac', 'geodesic', 'zero']) shall be tuned.
    add_noise: bool
        [OPTIONAL] Adds Gaussian noise to the mesh data.
    segmentation: bool
        [OPTIONAL] Whether to train the IMCNN for a shape segmentation problem instead of the shape correspondence
        problem.
    save_coordinates: bool
        [OPTIONAL] Whether to save the vertex coordinates in the dataset.
    epochs: int
        [OPTIONAL] Maximal amount of training epochs.
    seeds: list
        [OPTIONAL] List of integers that represent seeds to be used for the experiments. The amount of seeds also
        determine how often the experiment is repeated.
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
            precomputed_gpc_radius=precomputed_gpc_radius,
            processes=processes,
            add_noise=add_noise,
            save_coordinates=save_coordinates
        )
    else:
        print(f"Found preprocess-results: '{preprocess_zip}'. Skipping preprocessing.")

    # Set signal dim
    signal_dim = 544 if compute_shot else 3

    # Set seeds
    if seeds is None:
        seeds = [10, 20, 30, 40, 50]

    for exp_number in range(len(seeds)):
        # Set seeds
        tf.random.set_seed(seeds[exp_number])
        np.random.seed(seeds[exp_number])

        # Set kernel size
        kernel_size = (n_radial, n_angular)

        ##########
        # MODEL 1
        ##########
        print("#############################################")
        print("RUN TRAINING OF GROUND TRUTH PREDICTING MODEL")
        print("#############################################")
        # Load data
        train_data = load_preprocessed_faust(preprocess_zip, signal_dim=signal_dim, kernel_size=kernel_size, set_type=3)

        # Train model
        imcnn = run_training(
            signal_dim,
            kernel_size,
            template_radius,
            layer_conf,
            model_variant,
            segmentation,
            init_lr,
            weight_decay,
            preprocess_zip,
            logging_dir,
            f"{exp_number}_GT",
            train_data,
            epochs
        )

        # Create new dataset
        imcnn_new_dataset(imcnn, preprocess_zip, new_dataset_path)
        preprocess_zip = f"{new_dataset_path}.zip"

        ##########
        # MODEL 2
        ##########
        print("#############################################")
        print("RUN TRAINING OF EVALUATION MODEL")
        print("#############################################")

        # Load data
        train_data = load_preprocessed_faust(preprocess_zip, signal_dim=signal_dim, kernel_size=kernel_size, set_type=0)
        val_data = load_preprocessed_faust(preprocess_zip, signal_dim=signal_dim, kernel_size=kernel_size, set_type=1)

        # Train model
        imcnn = run_training(
            signal_dim,
            kernel_size,
            template_radius,
            layer_conf,
            model_variant,
            segmentation,
            init_lr,
            weight_decay,
            preprocess_zip,
            logging_dir,
            f"{exp_number}_EVALUATION",
            train_data,
            epochs,
            val_data
        )

        # Load dataset
        test_dataset = faust_generator(
            preprocess_zip, set_type=2, only_signal=False, return_coordinates=True
        )

        # Configure statistics
        loss = keras.metrics.SparseCategoricalCrossentropy()
        acc = keras.metrics.SparseCategoricalAccuracy()

        # Test loop
        for (signal, bc, coord), gt in test_dataset:
            pred = imcnn([signal, bc])
            loss.update_state(gt, pred)
            acc.update_state(gt, pred)
            sys.stdout.write(f"\rTest accuracy: {acc.result()} - Test loss: {loss.result()}")
