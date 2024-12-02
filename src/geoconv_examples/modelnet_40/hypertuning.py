from geoconv_examples.modelnet_40.classifier import ModelNetClf
from geoconv_examples.modelnet_40.dataset import load_preprocessed_modelnet

import tensorflow as tf
import keras_tuner as kt
import os


def hyper_tuning(dataset_path,
                 logging_dir,
                 template_configuration,
                 neighbors_for_lrf,
                 modelnet10=True,
                 gen_info_file=None,
                 batch_size=4,
                 rotation_delta=1,
                 pooling="max"):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    n_radial, n_angular, template_radius = template_configuration

    def build_hypermodel(hp):
        # Configure classifier
        imcnn = ModelNetClf(
            neighbors_for_lrf=neighbors_for_lrf,
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=template_radius,
            isc_layer_dims=[16 for _ in range(4)] + [32 for _ in range(2)],
            modelnet10=modelnet10,
            variant="dirac",
            rotation_delta=rotation_delta,
            dropout_rate=0.,
            pooling=hp.Choice("pooling", values=["avg", "cov", "max"]),
            noise_stddev=hp.Float("noise_stddev", min_value=0.00001, max_value=0.001)
        )

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum_over_batch_size")
        opt = tf.keras.optimizers.AdamW(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=hp.Float("learning_rate", min_value=0.0001, max_value=0.1),
                decay_steps=hp.Int("decay_steps", min_value=1, max_value=12305),
                decay_rate=hp.Float("lr_exp_decay", min_value=0.1, max_value=0.99999),
                staircase=hp.Boolean("staircase")
            ),
            weight_decay=hp.Float("weight_decay", min_value=0.0001, max_value=0.99999)
        )
        imcnn.compile(optimizer=opt, loss=loss, metrics=["accuracy"], run_eagerly=True)

        return imcnn

    tuner = kt.BayesianOptimization(
        hypermodel=build_hypermodel,
        objective="val_accuracy",
        max_trials=10_000,
        num_initial_points=15,
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
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, min_delta=0.01)
    tuner.search(x=train_data, validation_data=test_data, epochs=8, callbacks=[stop])

    # Print best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:")
    for key, value in best_hp.values.items():
        print(key, value)
