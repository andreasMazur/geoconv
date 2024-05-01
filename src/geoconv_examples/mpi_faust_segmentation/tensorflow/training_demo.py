from geoconv_examples.mpi_faust.tensorflow.model import Imcnn
from geoconv_examples.mpi_faust.tensorflow.faust_data_set import load_preprocessed_faust
from geoconv_examples.mpi_faust_segmentation.tensorflow.convert_dataset import convert_dataset
from geoconv_examples.mpi_faust_segmentation.tensorflow.convert_dataset import convert_dataset_deepview

from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import keras
import sys
import os


def visualize_csv(csv_path, figure_name="training_statistics"):
    """Visualize training statistics

    TODO: Add docstring

    Parameters
    ----------
    csv_path
    figure_name
    """
    csv = pd.read_csv(csv_path)

    # Configure plot
    n_rows, n_cols, cm = 2, 1, 1/2.54
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12*cm, 12.2*cm), sharex=True)
    for j in range(n_rows):
        axs[j].grid()

    # Configure axes
    axs[0].plot(csv["loss"], label="loss")
    axs[0].plot(csv["val_loss"], label="val_loss")
    axs[0].set_ylabel("Loss")

    axs[1].plot(csv["sparse_categorical_accuracy"])
    axs[1].plot(csv["val_sparse_categorical_accuracy"])
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Sparse Categorical Accuracy")

    plt.savefig(f"{figure_name}.svg", bbox_inches="tight")
    plt.show()


def train_model(training_data,
                validation_data,
                test_data,
                adaptation_data,
                n_radial=5,
                n_angular=8,
                template_radius=0.027744965069279016,
                logging_dir="./imcnn_training_logs",
                init_lr=0.00165,
                weight_decay=0.005,
                layer_conf=None,
                model_variant="dirac",
                segmentation=False,
                epochs=200,
                seeds=None):
    """Trains one singular IMCNN

    Parameters
    ----------
    training_data: TODO
    validation_data: TODO
    test_data: TODO
    adaptation_data: TODO
    n_radial: int
        [REQUIRED FOR PRE-PROCESSING] The amount of radial coordinates for the template.
    n_angular: int
        [REQUIRED FOR PRE-PROCESSING] The amount of angular coordinates for the template.
    template_radius: float
        [OPTIONAL] The template radius of the ISC-layer (the one used during preprocessing, defaults to radius for FAUST
        data set).
    logging_dir: str
        [OPTIONAL] The path to the folder where logs will be stored
    init_lr: float
        [OPTIONAL] Initial learning rate.
    weight_decay: float
        [OPTIONAL] Weight decay.
    layer_conf: list
        [OPTIONAL] List of tuples: The first entry references the output dimensions of the i-th ISC-layer, The second
        entry references of skips between each rotation while computing the convolution (rotation delta).
    model_variant: str
        [OPTIONAL] Which model variant (['dirac', 'geodesic', 'zero']) shall be tuned.
    segmentation: bool
        [OPTIONAL] Whether to train the IMCNN for a shape segmentation problem instead of the shape correspondence
        problem.
    epochs: int
        [OPTIONAL] Maximal amount of training epochs.
    seeds: list
        [OPTIONAL] List of integers that represent seeds to be used for the experiments. The amount of seeds also
        determine how often the experiment is repeated.

    Returns
    -------
    tuple:
        The test accuracy and loss.
    """
    # Set seeds
    if seeds is None:
        seeds = [10, 20, 30, 40, 50]

    test_accuracies, test_losses = [], []
    for exp_number in range(len(seeds)):
        svg_file_name = f"{logging_dir}/training_{exp_number}.log"
        if Path(svg_file_name).is_file():
            print(f"Found {svg_file_name}: Skipping this experiment iteration.")
            continue

        # Set seeds
        tf.random.set_seed(seeds[exp_number])
        np.random.seed(seeds[exp_number])

        # Set kernel size
        kernel_size = (n_radial, n_angular)

        # Define and compile model
        imcnn = Imcnn(
            signal_dim=544,
            kernel_size=kernel_size,
            template_radius=template_radius,
            layer_conf=layer_conf,
            variant=model_variant,
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
        imcnn.normalize.build(tf.TensorShape([6890, 544]))
        imcnn.normalize.adapt(adaptation_data)
        print("Done.")

        # Build model
        imcnn([tf.random.uniform(shape=(6890, 544)), tf.random.uniform(shape=(6890,) + kernel_size + (3, 2))])
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
        imcnn.fit(x=training_data, callbacks=[stop, tb, csv], validation_data=validation_data, epochs=epochs)
        imcnn.save(f"{logging_dir}/saved_imcnn_{exp_number}")

        # Configure statistics
        loss = keras.metrics.SparseCategoricalCrossentropy()
        acc = keras.metrics.SparseCategoricalAccuracy()

        # Test loop
        acc_value, loss_value = -1., -1.
        for (signal, bc), gt in test_data:
            pred = imcnn([signal, bc])

            # Statistics
            loss.update_state(gt, pred)
            acc.update_state(gt, pred)
            acc_value = acc.result()
            loss_value = loss.result()

            sys.stdout.write(f"\rTest accuracy: {acc_value} - Test loss: {loss_value}")

        # Log final accuracy and loss values of test phase
        test_accuracies.append(float(acc_value))
        test_losses.append(float(loss_value))

        # Visualize training results
        visualize_csv(svg_file_name, figure_name=f"{logging_dir}/training_statistics_{exp_number}")

    return test_accuracies, test_losses


def run_experiment(registration_path,
                   old_dataset_path,
                   bbox_segmentation_ds_path,
                   csv_path,
                   deepview_segmentation_ds_path,
                   logging_dir):
    """Creates the datasets and starts the training runs.

    Parameters
    ----------
    registration_path: str
        The path to the FAUST registration files
    old_dataset_path: str
        The path to the originally pre-processed FAUST dataset
    bbox_segmentation_ds_path: str
        The path to the bounding-box segmented dataset
    csv_path: str
        The path to the CSV-file that contains the DeepView-corrections
    deepview_segmentation_ds_path: str
        The path to the DeepView-corrected dataset
    logging_dir: str
        The path to the logging directory
    """
    # -- Training configuration --
    n_radial = 5
    n_angular = 8
    kernel_size = (n_radial, n_angular)
    precomputed_gpc_radius = 0.036993286759038686
    template_radius = precomputed_gpc_radius * 0.75
    layer_conf = [(96, 1)]
    init_lr = 0.00165
    weight_decay = 0.005
    model_variant = "dirac"
    epochs = 10
    seeds = [x * 10 for x in range(30)]
    # ----------------------------

    # Create logging dir
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Use bounding-boxes to convert shape-correspondence to segmentation labels
    converted_zip_bbox = f"{bbox_segmentation_ds_path}.zip"
    if not Path(converted_zip_bbox).is_file():
        convert_dataset(
            registration_path=registration_path,
            old_dataset_path=old_dataset_path,
            new_dataset_path=bbox_segmentation_ds_path
        )
    else:
        print(f"Found converted dataset file: '{converted_zip_bbox}'. Skipping preprocessing.")

    # Integrate DeepView-corrected labels into the dataset
    converted_zip_deepview = f"{deepview_segmentation_ds_path}.zip"
    if not Path(converted_zip_deepview).is_file():
        convert_dataset_deepview(
            csv_path=csv_path,
            old_dataset_path=converted_zip_bbox,
            new_dataset_path=deepview_segmentation_ds_path
        )
    else:
        print(f"Found converted dataset file: '{converted_zip_deepview}'. Skipping preprocessing.")

    # Use corrected data as test data
    val_data = load_preprocessed_faust(converted_zip_deepview, signal_dim=544, kernel_size=kernel_size, set_type=1)
    test_data = load_preprocessed_faust(converted_zip_deepview, signal_dim=544, kernel_size=kernel_size, set_type=2)

    # Start training runs
    test_accuracies, test_losses, sub_logging_dirs = [], [], ["bbox_approach", "deepview_approach"]
    for idx, zip_file in enumerate([converted_zip_bbox, converted_zip_deepview]):
        train_data = load_preprocessed_faust(zip_file, signal_dim=544, kernel_size=kernel_size, set_type=0)
        adaptation_data = load_preprocessed_faust(
            zip_file, signal_dim=544, kernel_size=kernel_size, set_type=0, only_signal=True
        )
        accuracies, losses = train_model(
            training_data=train_data,
            validation_data=val_data,
            test_data=test_data,
            adaptation_data=adaptation_data,
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=template_radius,
            logging_dir=f"{logging_dir}/{sub_logging_dirs[idx]}",
            layer_conf=layer_conf,
            init_lr=init_lr,
            weight_decay=weight_decay,
            model_variant=model_variant,
            segmentation=True,
            epochs=epochs,
            seeds=seeds
        )
        test_accuracies.append(accuracies)
        test_losses.append(losses)

    # Compute and store Mann-Whitney U test
    acc_test_statistic, acc_p_value = sp.stats.ranksums(x=test_accuracies[0], y=test_accuracies[1])
    loss_test_statistic, loss_p_value = sp.stats.ranksums(x=test_losses[0], y=test_losses[1])
    mwutest_file_name = f"{logging_dir}/mann_whitney_u_test.txt"
    if not os.path.exists(mwutest_file_name):
        os.mknod(mwutest_file_name)
    with open(mwutest_file_name, "w") as f:
        f.write("### MANN-WHITNEY-U TEST ###\n")
        f.write(f"Test accuracy statistic: {acc_test_statistic}\n")
        f.write(f"Test accuracy p-value: {acc_p_value}\n")
        f.write(f"Test loss statistic: {loss_test_statistic}\n")
        f.write(f"Test loss p-value: {loss_p_value}\n")
        f.write("###########################\n")
        f.write(f"Captured test accuracies (bbox-approach): {test_accuracies[0]}\n")
        f.write(f"Captured test accuracies (deepview-approach): {test_accuracies[1]}\n")
        f.write("###########################\n")
        f.write(f"Captured test loss (bbox-approach): {test_losses[0]}\n")
        f.write(f"Captured test loss (deepview-approach): {test_losses[1]}\n")
        f.write("###########################\n")
