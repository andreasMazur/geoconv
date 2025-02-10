from geoconv_examples.modelnet_40.classifier import ModelNetClf, WarmupAndExpDecay
from geoconv_examples.modelnet_40.dataset import load_preprocessed_modelnet

import tensorflow as tf
import keras_tuner as kt
import os


def hyper_tuning(dataset_path,
                 logging_dir,
                 template_configuration,
                 neighbors_for_lrf,
                 projection_neighbors,
                 modelnet10=True,
                 gen_info_file=None,
                 batch_size=4,
                 rotation_delta=1,
                 variant="dirac",
                 isc_layer_conf=None,
                 azimuth_bins=8,
                 elevation_bins=2,
                 radial_bins=2,
                 histogram_bins=11,
                 sphere_radius=0.,
                 exp_lambda=2.0,
                 shift_angular=True):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    n_radial, n_angular, template_radius = template_configuration

    def build_hypermodel(hp):
        # Configure classifier
        imcnn = ModelNetClf(
            neighbors_for_lrf=neighbors_for_lrf,
            projection_neighbors=projection_neighbors,
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=template_radius,
            isc_layer_conf=isc_layer_conf,
            modelnet10=modelnet10,
            variant=variant,
            rotation_delta=rotation_delta,
            azimuth_bins=azimuth_bins,
            elevation_bins=elevation_bins,
            radial_bins=radial_bins,
            histogram_bins=histogram_bins,
            sphere_radius=sphere_radius,
            dropout_rate=0.,
            exp_lambda=exp_lambda,
            shift_angular=shift_angular,
            time=hp.Float("time", min_value=1e-6, max_value=5.),
            iterations=hp.Float("iterations", min_value=1, max_value=10)
        )

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum_over_batch_size")
        opt = tf.keras.optimizers.AdamW(
            learning_rate=WarmupAndExpDecay(
                initial_learning_rate=hp.Float("init_lr", min_value=0.0, max_value=0.001294820592282788 * 1.25),
                decay_steps=2461,  # One epoch
                decay_rate=hp.Float("weight_decay", min_value=0.0, max_value=1.0),
                warmup_steps=2461  # One epoch
            ),
            weight_decay=0.09246487756417879,
            beta_1=0.9,
            beta_2=0.999
        )

        imcnn.compile(optimizer=opt, loss=loss, metrics="accuracy", run_eagerly=True)

        return imcnn

    tuner = kt.BayesianOptimization(
        hypermodel=build_hypermodel,
        objective=kt.Objective(name="val_loss", direction="min"),
        max_trials=10_000,
        num_initial_points=6,
        directory=logging_dir,
        project_name="modelnet_40_hyper_tuning",
        tune_new_entries=True,
        allow_new_entries=True
    )

    # Setup datasets
    train_data = load_preprocessed_modelnet(
        dataset_path,
        set_type="train",
        modelnet10=modelnet10,
        gen_info_file=f"{logging_dir}/{gen_info_file}",
        batch_size=batch_size
    )
    test_data = load_preprocessed_modelnet(
        dataset_path,
        set_type="test",
        modelnet10=modelnet10,
        gen_info_file=f"{logging_dir}/test_{gen_info_file}",
        batch_size=batch_size
    )

    # Start hyperparameter tuning
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, min_delta=0.01)
    tuner.search(x=train_data, validation_data=test_data, epochs=20, callbacks=[stop])

    # Print best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:")
    for key, value in best_hp.values.items():
        print(key, value)
