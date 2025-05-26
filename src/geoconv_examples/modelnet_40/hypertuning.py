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
                 exp_lambda=3.0,
                 shift_angular=True,
                 dropout_rate=0.0):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    n_radial, n_angular = template_configuration

    def build_hypermodel(hp):
        # Configure classifier
        imcnn = ModelNetClf(
            kernel=variant,
            pooling=pooling,
            neighbors_for_lrf=neighbors_for_lrf,
            projection_neighbors=projection_neighbors,
            azimuth_bins=azimuth_bins,
            elevation_bins=elevation_bins,
            radial_bins=radial_bins,
            histogram_bins=histogram_bins,
            sphere_radius=sphere_radius,
            n_radial=n_radial,
            n_angular=n_angular,
            exp_lambda=exp_lambda,
            shift_angular=shift_angular,
            template_scale=1.0,
            isc_layer_conf=isc_layer_conf,
            rotation_delta=rotation_delta,
            dropout_rate=dropout_rate,
            l1_reg_strength=hp.Float("l1_reg", min_value=0.037 / 2, max_value=0.037 * 2),
            l2_reg_strength=0.0,
            modelnet10=True
        )
        imcnn.adapt(
            adapt_data=load_preprocessed_modelnet(
                dataset_path,
                set_type="train",
                modelnet10=True,
                gen_info_file=f"{logging_dir}/{gen_info_file}",
                batch_size=batch_size
            )
        )

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum_over_batch_size")
        opt = tf.keras.optimizers.Adam(
            learning_rate=hp.Float("initial_lr", min_value=0.0038 / 2, max_value=0.0038 * 2),
            # learning_rate=WarmupAndExpDecay(
            #     initial_learning_rate=hp.Float("initial_lr", min_value=0.0004, max_value=0.004),
            #     decay_steps=998,
            #     decay_rate=hp.Float("lr_decay", min_value=0.65, max_value=1.0),
            #     warmup_steps=998
            # ),
            beta_1=0.9,
            beta_2=0.999
        )
        imcnn.compile(optimizer=opt, loss=loss, metrics="accuracy", run_eagerly=True)

        return imcnn

    tuner = kt.BayesianOptimization(
        hypermodel=build_hypermodel,
        objective=kt.Objective(name="val_accuracy", direction="max"),
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
        modelnet10=True,
        gen_info_file=f"{logging_dir}/{gen_info_file}",
        batch_size=batch_size
    )
    test_data = load_preprocessed_modelnet(
        dataset_path,
        set_type="test",
        modelnet10=True,
        gen_info_file=f"{logging_dir}/test_{gen_info_file}",
        batch_size=batch_size
    )

    # Start hyperparameter tuning
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, min_delta=0.001)
    tuner.search(x=train_data, validation_data=test_data, epochs=50, callbacks=[stop])

    # Print best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:")
    for key, value in best_hp.values.items():
        print(key, value)
