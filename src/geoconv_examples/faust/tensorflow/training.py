from geoconv.tensorflow.layers import ConvDirac, ConvGeodesic, ConvZero, AngularMaxPooling
from geoconv.utils.data_generator import read_template_configurations
from geoconv.utils.princeton_benchmark import princeton_benchmark
from geoconv_examples.faust.tensorflow.classifier import build_faust_classifier
from geoconv_examples.faust.tensorflow.dataset import load_preprocessed_faust

import tensorflow as tf
import os


def training(dataset_path,
             logging_dir,
             reference_mesh_path,
             template_configurations=None,
             variant=None,
             processes=1,
             isc_layer_dims=None,
             learning_rate=0.00165,
             gen_info_file=None,
             rotation_delta=None,
             batch_size=1):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Prepare template configurations
    if template_configurations is None:
        template_configurations = read_template_configurations(dataset_path)

    # Set filename for generator
    if gen_info_file is None:
        gen_info_file = "generator_info.json"

    # Run experiments
    for (n_radial, n_angular, template_radius) in template_configurations:
        # Load data
        train_data = load_preprocessed_faust(
            dataset_path,
            n_radial,
            n_angular,
            template_radius,
            is_train=True,
            gen_info_file=f"{logging_dir}/{gen_info_file}",
            batch_size=batch_size
        )
        test_data = load_preprocessed_faust(
            dataset_path,
            n_radial,
            n_angular,
            template_radius,
            is_train=False,
            gen_info_file=f"{logging_dir}/test_{gen_info_file}",
            batch_size=batch_size
        )

        # Build model
        imcnn = build_faust_classifier(
            variant=variant,
            n_radial=n_radial,
            n_angular=n_angular,
            isc_layer_dims=isc_layer_dims,
            template_radius=template_radius,
            rotation_delta=n_angular if rotation_delta is None else rotation_delta
        )

        # Compile model
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        imcnn.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        imcnn.summary()

        # Define callbacks
        exp_number = f"{n_radial}_{n_angular}_{template_radius}"
        csv_file_name = f"{logging_dir}/training_{exp_number}.log"
        csv = tf.keras.callbacks.CSVLogger(csv_file_name)
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
        tb = tf.keras.callbacks.TensorBoard(
            log_dir=f"{logging_dir}/tensorboard_{exp_number}",
            histogram_freq=1,
            write_graph=False,
            write_steps_per_second=True,
            update_freq="epoch",
            profile_batch=(1, 80)
        )

        saving_path = f"{logging_dir}/saved_imcnn_{exp_number}.keras"
        save = tf.keras.callbacks.ModelCheckpoint(
            filepath=saving_path,
            monitor="val_loss",
            save_best_only=True,
            save_freq="epoch"
        )

        # Train model
        imcnn.fit(x=train_data, callbacks=[stop, tb, csv, save], validation_data=test_data, epochs=200)

        # Load best model
        best_imcnn = tf.keras.models.load_model(
            saving_path,
            custom_objects={
                "ConvDirac": ConvDirac,
                "ConvGeodesic": ConvGeodesic,
                "ConvZero": ConvZero,
                "AngularMaxPooling": AngularMaxPooling
            }
        )

        # Evaluate model with Princeton benchmark
        test_data = load_preprocessed_faust(
            dataset_path,
            n_radial,
            n_angular,
            template_radius,
            is_train=False,
            gen_info_file=f"{logging_dir}/test_{gen_info_file}",
            batch_size=1
        )
        princeton_benchmark(
            imcnn=best_imcnn,
            test_dataset=test_data,
            ref_mesh_path=reference_mesh_path,
            normalize=True,
            file_name=f"{logging_dir}/model_benchmark_{exp_number}",
            processes=processes,
            geodesic_diameter=2.2093810817030244
        )
