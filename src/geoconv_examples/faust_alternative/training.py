from geoconv.utils.princeton_benchmark import princeton_benchmark
from geoconv_examples.faust.classifer import FaustVertexClassifier, AMOUNT_VERTICES, SIG_DIM
from geoconv_examples.faust_alternative.classifer import MultiBranchClf
from geoconv_examples.faust_alternative.dataset import load_preprocessed_faust

import tensorflow as tf
import os


def training(dataset_path,
             logging_dir,
             reference_mesh_path,
             temp_conf_1,
             temp_conf_2,
             temp_conf_3,
             variant=None,
             processes=1,
             isc_layer_dims=None,
             learning_rate=0.00165,
             gen_info_file=None,
             rotation_delta=1,
             batch_size=1,
             middle_layer_dim=1024,
             dropout_rate=0.3,
             output_rotation_delta=1,
             l1_reg=0.3):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)
    # Set filename for generator
    if gen_info_file is None:
        gen_info_file = "generator_info.json"

    # Load data
    train_data = load_preprocessed_faust(
        dataset_path,
        temp_conf_1=temp_conf_1,
        temp_conf_2=temp_conf_2,
        temp_conf_3=temp_conf_3,
        is_train=True,
        gen_info_file=f"{logging_dir}/{gen_info_file}",
        batch_size=batch_size
    )
    test_data = load_preprocessed_faust(
        dataset_path,
        temp_conf_1=temp_conf_1,
        temp_conf_2=temp_conf_2,
        temp_conf_3=temp_conf_3,
        is_train=False,
        gen_info_file=f"{logging_dir}/test_{gen_info_file}",
        batch_size=batch_size
    )

    # Build model
    imcnn = MultiBranchClf(
        temp_radius_1=temp_conf_1[2],
        temp_radius_2=temp_conf_2[2],
        temp_radius_3=temp_conf_3[2],
        isc_layer_dims=isc_layer_dims,
        middle_layer_dim=middle_layer_dim,
        variant=variant,
        normalize_input=True,
        rotation_delta=rotation_delta,
        dropout_rate=dropout_rate,
        output_rotation_delta=output_rotation_delta,
        l1_reg=l1_reg
    )
    imcnn.build(
        input_shape=[
            tf.TensorShape([None, AMOUNT_VERTICES, SIG_DIM]),
            tf.TensorShape([None, AMOUNT_VERTICES, temp_conf_1[0], temp_conf_1[1], 3, 2]),
            tf.TensorShape([None, AMOUNT_VERTICES, temp_conf_2[0], temp_conf_2[1], 3, 2]),
            tf.TensorShape([None, AMOUNT_VERTICES, temp_conf_3[0], temp_conf_3[1], 3, 2])
        ]
    )
    # Adapt normalization: Normalize each vertex-feature-dimension (axis=-1) with individual mean and variance
    imcnn.branch_1.normalize.adapt(
        load_preprocessed_faust(
            dataset_path,
            temp_conf_1=temp_conf_1,
            temp_conf_2=temp_conf_2,
            temp_conf_3=temp_conf_3,
            is_train=True,
            gen_info_file=f"{logging_dir}/{gen_info_file}",
            only_signal=True
        )
    )
    imcnn.branch_2.normalize.adapt(
        load_preprocessed_faust(
            dataset_path,
            temp_conf_1=temp_conf_1,
            temp_conf_2=temp_conf_2,
            temp_conf_3=temp_conf_3,
            is_train=True,
            gen_info_file=f"{logging_dir}/{gen_info_file}",
            only_signal=True
        )
    )
    imcnn.branch_3.normalize.adapt(
        load_preprocessed_faust(
            dataset_path,
            temp_conf_1=temp_conf_1,
            temp_conf_2=temp_conf_2,
            temp_conf_3=temp_conf_3,
            is_train=True,
            gen_info_file=f"{logging_dir}/{gen_info_file}",
            only_signal=True
        )
    )

    # Compile model
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.AdamW(
        # learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=learning_rate,
        #     decay_steps=500,
        #     decay_rate=0.99999
        # ),
        learning_rate=learning_rate,
        weight_decay=0.005
    )
    imcnn.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    imcnn.summary()

    # Define callbacks
    csv_file_name = f"{logging_dir}/training.log"
    csv = tf.keras.callbacks.CSVLogger(csv_file_name)
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=f"{logging_dir}/tensorboard",
        histogram_freq=1,
        write_graph=False,
        write_steps_per_second=True,
        update_freq="epoch",
        profile_batch=(1, 80)
    )
    save = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{logging_dir}/saved_imcnn",
        monitor="val_loss",
        save_best_only=True,
        save_freq="epoch"
    )

    # Train model
    imcnn.fit(x=train_data, callbacks=[stop, tb, csv, save], validation_data=test_data, epochs=200)

    # Load best model
    imcnn_best = tf.keras.models.load_model(f"{logging_dir}/saved_imcnn")
    imcnn = FaustVertexClassifier(
        temp_radius_1=temp_conf_1[2],
        temp_radius_2=temp_conf_2[2],
        temp_radius_3=temp_conf_3[2],
        isc_layer_dims=isc_layer_dims,
        middle_layer_dim=middle_layer_dim,
        variant=variant,
        normalize_input=True,
        rotation_delta=rotation_delta
    )
    imcnn.build(
        input_shape=[
            tf.TensorShape([None, AMOUNT_VERTICES, SIG_DIM]),
            tf.TensorShape([None, AMOUNT_VERTICES, temp_conf_1[0], temp_conf_1[1], 3, 2]),
            tf.TensorShape([None, AMOUNT_VERTICES, temp_conf_2[0], temp_conf_2[1], 3, 2]),
            tf.TensorShape([None, AMOUNT_VERTICES, temp_conf_3[0], temp_conf_3[1], 3, 2])
        ]
    )
    imcnn.set_weights(imcnn_best.get_weights())

    # Evaluate model with Princeton benchmark
    test_data = load_preprocessed_faust(
        dataset_path,
        temp_conf_1=temp_conf_1,
        temp_conf_2=temp_conf_2,
        temp_conf_3=temp_conf_3,
        is_train=False,
        gen_info_file=f"{logging_dir}/test_{gen_info_file}",
        batch_size=1
    )
    princeton_benchmark(
        imcnn=imcnn,
        test_dataset=test_data,
        ref_mesh_path=reference_mesh_path,
        normalize=True,
        file_name=f"{logging_dir}/model_benchmark",
        processes=processes,
        geodesic_diameter=2.2093810817030244
    )