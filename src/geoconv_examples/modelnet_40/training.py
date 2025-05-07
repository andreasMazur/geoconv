from geoconv.tensorflow.layers import BarycentricCoordinates
from geoconv_examples.modelnet_40.classifier import ModelNetClf
from geoconv_examples.modelnet_40.dataset import load_preprocessed_modelnet

import os
import tensorflow as tf
import json
import gc
import numpy as np
import random


def model_configuration(n_radial,
                        n_angular,
                        bc_adapt_data,
                        template_scale,
                        isc_layer_conf,
                        neighbors_for_lrf,
                        projection_neighbors,
                        modelnet10,
                        kernel,
                        rotation_delta,
                        pooling,
                        exp_lambda,
                        shift_angular):
    # Determine template-radius
    template_radius = BarycentricCoordinates(
        n_radial=n_radial,
        n_angular=n_angular,
        neighbors_for_lrf=neighbors_for_lrf,
        projection_neighbors=projection_neighbors
    ).adapt(
        data=bc_adapt_data,
        template_scale=template_scale,
        exp_lambda=exp_lambda,
        shift_angular=shift_angular
    ).numpy()

    # Define model
    imcnn = ModelNetClf(
        n_radial=n_radial,
        n_angular=n_angular,
        isc_layer_conf=isc_layer_conf,
        template_radius=float(template_radius),
        neighbors_for_lrf=neighbors_for_lrf,  # Set higher than projection-neighbors
        projection_neighbors=projection_neighbors,
        modelnet10=modelnet10,
        kernel=kernel,
        rotation_delta=rotation_delta,
        pooling=pooling,
        exp_lambda=exp_lambda,
        shift_angular=shift_angular,
        azimuth_bins=8,
        elevation_bins=6,
        radial_bins=2,
        histogram_bins=6,
        sphere_radius=0.
    )
    imcnn.bc_layer.adapt(template_radius=template_radius, exp_lambda=exp_lambda, shift_angular=shift_angular)

    # Define loss and optimizer
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum_over_batch_size")
    opt = tf.keras.optimizers.AdamW(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0020618479317126375,
            decay_steps=2461,
            decay_rate=0.8762837040974372,
            staircase=False
        ),
        weight_decay=0.019081993138727875
    )

    # Compile the model
    imcnn.compile(optimizer=opt, loss=loss, metrics="accuracy", run_eagerly=True)

    # Build model
    imcnn(tf.random.uniform(shape=[1, 1024, 3]), training=False)
    imcnn.summary()

    return imcnn


def training(dataset_path,
             logging_dir,
             template_resolution,
             gen_info_file,
             batch_size=1,
             kernel=None,
             exp_lambda=3.0,
             shift_angular=True,
             pooling="avg",
             isc_layer_conf=None,
             projection_neighbor_list=None,
             coefficient_list=None,
             neighbors_for_lrf_list=None,
             rotation_delta_list=None,
             n_repetitions=3):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Initialize training summary
    repetitions_summary_path = f"{logging_dir}/repetitions_summary.json"
    if os.path.isfile(repetitions_summary_path):
        training_summary = json.load(open(f"{logging_dir}/repetitions_summary.json", "r"))
    else:
        training_summary = {}

    if isc_layer_conf is None:
        isc_layer_conf = [32, 32]
    n_radial, n_angular = template_resolution

    for repetition in range(n_repetitions):
        # Initialize default testing-values
        if projection_neighbor_list is None:
            projection_neighbor_list = [8, 12, 16]
        if coefficient_list is None:
            coefficient_list = [0.75, 1.0, 1.25]
        if rotation_delta_list is None:
            rotation_delta_list = list(range(1, n_angular))

        for projection_neighbors in projection_neighbor_list:
            if neighbors_for_lrf_list is None:
                neighbors_for_lrf_list = [i for i in range(projection_neighbors, 20, 4)]
            for template_scale in coefficient_list:
                for neighbors_for_lrf in neighbors_for_lrf_list:
                    for rotation_delta in rotation_delta_list:
                        experiment_id = (
                            f"repetition_{repetition}_"
                            f"proj_neigh_{projection_neighbors}_"
                            f"temp_scale_{template_scale}_"
                            f"neighbors_for_lrf_{neighbors_for_lrf}_"
                            f"rotation_delta_{rotation_delta}"
                        )
                        # Skip experiment if it's already stored in training summary
                        if experiment_id in training_summary.keys():
                            print(f"Experiment '{experiment_id}' already done, skipping...")
                            continue
                        print(f"Running experiment: '{experiment_id}'...")

                        training_summary[experiment_id] = []

                        rep_logging_dir = f"{logging_dir}/{repetition}/{experiment_id}"
                        os.makedirs(f"{rep_logging_dir}", exist_ok=True)

                        # Set seeds
                        np.random.seed(repetition)
                        tf.random.set_seed(repetition)
                        random.seed(repetition)

                        # Get classification model
                        imcnn = model_configuration(
                            n_radial=n_radial,
                            n_angular=n_angular,
                            bc_adapt_data=load_preprocessed_modelnet(
                                dataset_path,
                                set_type="train",
                                modelnet10=True,
                                gen_info_file=gen_info_file,
                                batch_size=1,
                                debug_data=False
                            ),
                            template_scale=template_scale,
                            isc_layer_conf=isc_layer_conf,
                            neighbors_for_lrf=neighbors_for_lrf,
                            projection_neighbors=projection_neighbors,
                            modelnet10=True,
                            kernel=kernel,
                            rotation_delta=rotation_delta,
                            pooling=pooling,
                            exp_lambda=exp_lambda,
                            shift_angular=shift_angular
                        )

                        # Define callbacks
                        csv_file_name = f"{rep_logging_dir}/{experiment_id}.log"
                        csv = tf.keras.callbacks.CSVLogger(csv_file_name)
                        stop = tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=10, min_delta=0.01, verbose=True
                        )
                        tb = tf.keras.callbacks.TensorBoard(
                            log_dir=f"{rep_logging_dir}/tensorboard_{experiment_id}",
                            histogram_freq=1,
                            write_graph=False,
                            write_steps_per_second=True,
                            update_freq="epoch",
                            profile_batch=(1, 200)
                        )

                        # Load data
                        train_data = load_preprocessed_modelnet(
                            dataset_path,
                            set_type="train",
                            modelnet10=True,
                            gen_info_file=gen_info_file,
                            batch_size=batch_size,
                            debug_data=False
                        )
                        test_data = load_preprocessed_modelnet(
                            dataset_path,
                            set_type="test",
                            modelnet10=True,
                            gen_info_file=gen_info_file,
                            batch_size=batch_size,
                            debug_data=False
                        )
                        save = tf.keras.callbacks.ModelCheckpoint(
                            filepath=f"{rep_logging_dir}/saved_imcnn_{experiment_id}.keras",
                            monitor="val_accuracy",
                            save_best_only=True,
                            save_freq="epoch"
                        )

                        # Train model
                        training_history = imcnn.fit(
                            x=train_data, callbacks=[stop, tb, csv, save], validation_data=test_data, epochs=200
                        )

                        # Collect training statistics
                        training_summary[experiment_id].append(max(training_history.history["val_accuracy"]))

                        # Free memory
                        gc.collect()
                        tf.keras.backend.clear_session()

                        with open(repetitions_summary_path, "w") as f:
                            json.dump(training_summary, f, indent=4)
