from geoconv.tensorflow.layers import ConvDirac, ConvGeodesic, ConvZero, BarycentricCoordinates
from geoconv.utils.data_generator import read_template_configurations
from geoconv.utils.prepare_logs import process_logs
from geoconv.tensorflow.layers import AngularMaxPooling
from geoconv_examples.mnist.tensorflow.dataset_projections import load_preprocessed_mnist_for_projections

import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import os


def build_mnist_classifier(variant, adapt_data, n_radial, n_angular, template_scale, rotation_delta, isc_layer_dims):
    if variant is None or variant == "dirac":
        layer_type = ConvDirac
    elif variant == "geodesic":
        layer_type = ConvGeodesic
    elif variant == "zero":
        layer_type = ConvZero
    else:
        raise RuntimeError("Select a layer type from: ['dirac', 'geodesic', 'zero']")

    # Configure BC-layer
    bc_layer = BarycentricCoordinates(
        n_radial=n_radial,
        n_angular=n_angular,
        neighbors_for_lrf=20,
        projection_neighbors=20,
    )
    template_radius = bc_layer.adapt(
        data=adapt_data,
        template_scale=template_scale,
    )
    print(f"### Template scale {template_scale} causes a template radius of: {template_radius} ###")

    # Point cloud to barycentric coordinates
    point_cloud_input = tf.keras.Input(shape=(28 * 28, 3), name="point_cloud_input", dtype=tf.float32)
    bc = bc_layer(point_cloud_input)

    # Fold image input
    image_input = tf.keras.Input(shape=(28 * 28, 1), name="image_input", dtype=tf.float32)
    signal = image_input
    for n in isc_layer_dims:
        signal = layer_type(
            amt_templates=n,
            template_radius=template_radius,
            activation="elu",
            rotation_delta=rotation_delta
        )([signal, bc])
        signal = AngularMaxPooling()(signal)
    signal = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")(signal)
    output = tf.keras.layers.Dense(10, activation="linear")(signal)
    imcnn = keras.Model(inputs=[image_input, point_cloud_input], outputs=output, name="mnist_model")
    return imcnn, template_radius


def training(logging_dir,
             k=5,
             template_configurations=None,
             bc_path=None,
             variant=None,
             batch_size=8,
             learning_rate=0.0017807,
             isc_layer_dims=None):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Setup default layer parameterization if not given
    if isc_layer_dims is None:
        isc_layer_dims = [128]

    # Prepare k-fold cross-validation
    splits = tfds.even_splits("all", n=k)

    # Prepare template configurations
    if template_configurations is None:
        assert isinstance(bc_path, str), "If 'template_configurations' is given, 'bc_path' must be be given."
        template_configurations = read_template_configurations(bc_path)

    # Run experiments
    for (n_radial, n_angular, template_scale) in template_configurations:
        csv_file_names = []
        for exp_no in range(len(splits)):
            # Load data
            train_data = load_preprocessed_mnist_for_projections(
                set_type=splits[:exp_no] + splits[exp_no+1:],
                batch_size=batch_size
            )
            val_data = load_preprocessed_mnist_for_projections(
                set_type=splits[exp_no],
                batch_size=batch_size
            )

            # Define and compile model
            imcnn, template_radius = build_mnist_classifier(
                variant=variant,
                adapt_data=load_preprocessed_mnist_for_projections(
                    set_type=splits[:exp_no] + splits[exp_no + 1:],
                    batch_size=1,
                    for_adaptation=True
                ),
                n_radial=n_radial,
                n_angular=n_angular,
                template_scale=template_scale,
                rotation_delta=n_angular,
                isc_layer_dims=isc_layer_dims
            )
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            imcnn.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=0.001 if learning_rate is None else learning_rate
                ),
                loss=loss,
                metrics=["accuracy"],
                run_eagerly=True
            )

            # Define callbacks
            exp_number = f"{exp_no}__{n_radial}_{n_angular}_{template_radius}"
            csv_file_name = f"{logging_dir}/training_{exp_number}.log"
            csv_file_names.append(csv_file_name)
            csv = keras.callbacks.CSVLogger(csv_file_name)
            tb = keras.callbacks.TensorBoard(
                log_dir=f"{logging_dir}/tensorboard_{exp_number}",
                histogram_freq=1,
                write_graph=False,
                write_steps_per_second=True,
                update_freq="epoch",
                profile_batch=(1, 100)
            )
            save = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{logging_dir}/saved_imcnn_{exp_number}.keras",
                monitor="val_loss",
                save_best_only=True,
                save_freq="epoch"
            )

            # Train model
            imcnn.fit(x=train_data, callbacks=[tb, csv, save], validation_data=val_data, epochs=100)

        # Process logs
        process_logs(
            csv_file_names, file_name=f"{logging_dir}/avg_training_{n_radial}_{n_angular}_{template_radius}.log"
        )
