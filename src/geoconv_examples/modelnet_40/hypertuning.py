from geoconv_examples.modelnet_40.classifier import ModelNetClf, WarmupAndExpDecay
from geoconv_examples.modelnet_40.dataset import load_preprocessed_modelnet, MN_CLASS_WEIGHTS

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
                 pooling="avg",
                 variant="dirac",
                 isc_layer_conf=None,
                 azimuth_bins=8,
                 elevation_bins=6,
                 radial_bins=2,
                 histogram_bins=6,
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
            pooling=pooling,
            azimuth_bins=azimuth_bins,
            elevation_bins=elevation_bins,
            radial_bins=radial_bins,
            histogram_bins=histogram_bins,
            sphere_radius=sphere_radius,
            dropout_rate=0.,
            exp_lambda=exp_lambda,
            shift_angular=shift_angular
        )

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum_over_batch_size")
        opt = tf.keras.optimizers.AdamW(
            learning_rate=WarmupAndExpDecay(
                initial_learning_rate=hp.Float("initial_lr", min_value=0.0, max_value=1.0),
                decay_steps=2461,
                decay_rate=hp.Float("lr_decay", min_value=0.9, max_value=1.0),
                warmup_steps=2461
            ),
            weight_decay=0.,  # hp.Float("weight_decay", min_value=0.0, max_value=1.0),
            beta_1=0.9,
            beta_2=0.999
        )

        imcnn.compile(optimizer=opt, loss=loss, metrics="accuracy", run_eagerly=True)

        return imcnn

    tuner = kt.BayesianOptimization(
        hypermodel=build_hypermodel,
        objective=kt.Objective(name="val_loss", direction="min"),
        max_trials=10_000,
        num_initial_points=12,
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
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, min_delta=0.001)
    tuner.search(x=train_data, validation_data=test_data, epochs=20, callbacks=[stop])

    # Print best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:")
    for key, value in best_hp.values.items():
        print(key, value)
