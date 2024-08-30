from geoconv.utils.data_generator import read_template_configurations
from geoconv_examples.faust.classifer import AMOUNT_VERTICES, SIG_DIM
from geoconv_examples.faust_embedding.dataset import load_faust_embedding_gen
from geoconv_examples.faust_embedding.siamese_net import SiameseIMCNN

import os
import tensorflow as tf


def cosine_loss(y_true, y_pred):
    return tf.reduce_sum(
        1 - tf.einsum("bni,bni->bn", tf.gather(y_pred[0], tf.cast(y_true, tf.int32), batch_dims=1), y_pred[1])
    )


def training(dataset_path,
             logging_dir,
             template_configurations=None,
             variant=None,
             isc_layer_dims=None,
             learning_rate=0.00165,
             gen_info_file=None,
             rotation_delta=1,
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
            train_data = load_faust_embedding_gen(
                dataset_path,
                n_radial,
                n_angular,
                template_radius,
                is_train=True,
                gen_info_file=f"{logging_dir}/{gen_info_file}",
                batch_size=batch_size
            )
            test_data = load_faust_embedding_gen(
                dataset_path,
                n_radial,
                n_angular,
                template_radius,
                is_train=False,
                gen_info_file=f"{logging_dir}/test_{gen_info_file}",
                batch_size=batch_size
            )

            # Define and compile model
            imcnn = SiameseIMCNN(
                template_radius,
                isc_layer_dims=isc_layer_dims,
                variant=variant,
                normalize_input=True,
                rotation_delta=rotation_delta
            )
            opt = tf.keras.optimizers.AdamW(
                learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=learning_rate,
                    decay_steps=500,
                    decay_rate=0.99999
                ),
                weight_decay=0.005
            )
            imcnn.compile(optimizer=opt, loss=cosine_loss, run_eagerly=True)
            imcnn.build(
                input_shape=[
                    tf.TensorShape([None, AMOUNT_VERTICES, SIG_DIM]),
                    tf.TensorShape([None, AMOUNT_VERTICES, n_radial, n_angular, 3, 2]),
                    tf.TensorShape([None, AMOUNT_VERTICES, SIG_DIM]),
                    tf.TensorShape([None, AMOUNT_VERTICES, n_radial, n_angular, 3, 2])
                ]
            )
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
            save = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{logging_dir}/saved_imcnn_{exp_number}",
                monitor="val_loss",
                save_best_only=True,
                save_freq="epoch"
            )

            # Train model
            imcnn.fit(x=train_data, callbacks=[stop, tb, csv, save], validation_data=test_data, epochs=200)
