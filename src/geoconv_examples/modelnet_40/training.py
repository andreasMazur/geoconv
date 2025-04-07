from geoconv_examples.modelnet_40.classifier import ModelNetClf
from geoconv_examples.modelnet_40.dataset import load_preprocessed_modelnet

import os
import tensorflow as tf
import json
import gc

TEMPLATE_SCALES = [0.75, 1.0, 1.25]
RHOS = [(10, 0.2342197448015213), (14, 0.27853941917419434),  (18, 0.3163464069366455)]


def model_configuration(neighbors_for_lrf,
                        projection_neighbors,
                        n_radial,
                        n_angular,
                        template_radius,
                        modelnet10,
                        kernel,
                        rotation_delta,
                        exp_lambda,
                        shift_angular,
                        isc_layer_conf,
                        pooling):
    # Define model
    imcnn = ModelNetClf(
        neighbors_for_lrf=neighbors_for_lrf,
        projection_neighbors=projection_neighbors,
        n_radial=n_radial,
        n_angular=n_angular,
        template_radius=template_radius,
        isc_layer_conf=isc_layer_conf,
        modelnet10=modelnet10,
        variant=kernel,
        rotation_delta=rotation_delta,
        pooling=pooling,
        azimuth_bins=8,
        elevation_bins=6,
        radial_bins=2,
        histogram_bins=6,
        sphere_radius=0.,
        exp_lambda=exp_lambda,
        shift_angular=shift_angular
    )

    # Define loss and optimizer
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum_over_batch_size")
    opt = tf.keras.optimizers.AdamW(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0007673139778927575,
            decay_steps=2461,
            decay_rate=0.7080237438158256,
            staircase=False
        ),
        weight_decay=0.019081993138727875
    )

    # Compile the model
    imcnn.compile(optimizer=opt, loss=loss, metrics="accuracy", run_eagerly=True)
    imcnn(tf.random.uniform(shape=[1, 1024, 3]), training=False)
    imcnn.summary()

    return imcnn


def training(dataset_path,
             logging_dir,
             template_resolution,
             neighbors_for_lrf=16,
             batch_size=1,
             kernel=None,
             rotation_delta=1,
             exp_lambda=2.0,
             shift_angular=True,
             pooling="avg"):
    os.makedirs(logging_dir, exist_ok=True)
    n_radial, n_angular = template_resolution
    training_summary = {}

    for (projection_neighbors, rho) in RHOS:
        for template_scale in TEMPLATE_SCALES:
            training_summary[rho * template_scale] = []
            for repetition in range(5):
                rep_logging_dir = f"{logging_dir}/{repetition}"
                os.makedirs(f"{rep_logging_dir}", exist_ok=True)

                # Get classification model
                imcnn = model_configuration(
                    neighbors_for_lrf=neighbors_for_lrf,
                    projection_neighbors=projection_neighbors,
                    n_radial=n_radial,
                    n_angular=n_angular,
                    template_radius=rho * template_scale,
                    modelnet10=True,
                    kernel=kernel,
                    rotation_delta=rotation_delta,
                    exp_lambda=exp_lambda,
                    shift_angular=shift_angular,
                    isc_layer_conf=[128, 128],
                    pooling=pooling
                )

                # Define callbacks
                exp_number = f"{n_radial}_{n_angular}_{template_scale}"
                csv_file_name = f"{rep_logging_dir}/training_{exp_number}.log"
                csv = tf.keras.callbacks.CSVLogger(csv_file_name)
                stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=0.01, verbose=True)
                tb = tf.keras.callbacks.TensorBoard(
                    log_dir=f"{rep_logging_dir}/tensorboard_{exp_number}",
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
                    gen_info_file=f"{rep_logging_dir}/generator_info.json",
                    batch_size=batch_size,
                    debug_data=True
                )
                test_data = load_preprocessed_modelnet(
                    dataset_path,
                    set_type="test",
                    modelnet10=True,
                    gen_info_file=f"{rep_logging_dir}/test_generator_info.json",
                    batch_size=batch_size,
                    debug_data=True
                )
                save = tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{rep_logging_dir}/saved_imcnn_{exp_number}.keras",
                    monitor="val_accuracy",
                    save_best_only=True,
                    save_freq="epoch"
                )

                # Train model
                training_history = imcnn.fit(
                    x=train_data, callbacks=[stop, tb, csv, save], validation_data=test_data, epochs=200
                )

                # Collect training statistics
                training_summary[rho * template_scale].append(max(training_history.history["val_accuracy"]))

                # Free memory
                gc.collect()
                tf.keras.backend.clear_session()

            with open(f"{logging_dir}/repetitions_summary.json", "w") as f:
                json.dump(training_summary, f, indent=4)
