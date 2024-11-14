from geoconv_examples.modelnet_40.dataset import load_preprocessed_modelnet, MODELNET_CLASSES
from geoconv_examples.modelnet_40.training import model_configuration

from sklearn import metrics

import tensorflow as tf
import numpy as np
import os
import json


def check_performance(model_path, model_config, data_path, logging_dir, modelnet10=False, gen_info_file=None):
    """Computes the Adjusted rand index (ARI) as well as the adjusted mutual info score (AMI)

    Parameters
    ----------
    model_path: str
        The path to the trained model that shall be analysed.
    model_config: dict
        A configuration dictionary that contains exactly the following keys and their respective values for the model
        that shall be loaded:
            - neighbors_for_lrf
            - n_radial
            - n_angular
            - template_radius
            - isc_layer_dims
            - variant
            - rotation_delta
            - pooling
    data_path: str
        The path to the data set.
    logging_dir: str
        The path to the directory where the log-files and results shall be stored.
    modelnet10: bool
        Whether to test on ModelNet10
    gen_info_file: str | None
        The path to where the generator-information file shall be stored.
    """
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    if gen_info_file is None:
        gen_info_file = "generator_info.json"

    # Load model
    imcnn = model_configuration(
        neighbors_for_lrf=model_config["neighbors_for_lrf"],
        n_radial=model_config["n_radial"],
        n_angular=model_config["n_angular"],
        template_radius=model_config["template_radius"],
        isc_layer_dims=model_config["isc_layer_dims"],
        modelnet10=modelnet10,
        learning_rate=0.,
        variant=model_config["variant"],
        rotation_delta=model_config["rotation_delta"],
        dropout_rate=0.,
        weight_decay=0.,
        pooling=model_config["pooling"],
        triplet_alpha=0.
    )
    imcnn.load_weights(model_path)

    test_data = load_preprocessed_modelnet(
        data_path,
        set_type="test",
        modelnet10=modelnet10,
        gen_info_file=f"{logging_dir}/test_{gen_info_file}",
        batch_size=1,
        debug_data=False
    )

    # Make predictions
    y_true = []
    y_pred = []
    dict_inverted = {k: v for v, k in MODELNET_CLASSES.items()}
    for idx, (coordinates, label) in enumerate(test_data):
        # Get certainty estimates:
        prediction, _ = imcnn(coordinates[:, :, 1, :])
        prediction = tf.nn.softmax(prediction[0])
        class_prediction = tf.argmax(prediction).numpy().astype(dtype=np.int32).tolist()
        y_true.append(label[0, 0].numpy().astype(np.int32).tolist())
        y_pred.append(class_prediction)

        # Console logging
        top_indices = tf.math.top_k(prediction, k=10)[1].numpy()
        print_dict = [dict_inverted[i] for i in top_indices]
        print_dict = {k: f"{v:.2f}" for k, v in zip(print_dict, prediction.numpy()[top_indices])}
        print(idx, dict_inverted[label[0, 0].numpy().astype(np.int32).tolist()], " | ", print_dict)

    # Compute scores
    ari_score = metrics.adjusted_rand_score(y_true, y_pred)
    ami_score = metrics.adjusted_mutual_info_score(y_true, y_pred)
    with open(f"{logging_dir}/performances_scores.json", "w") as properties_file:
        json.dump(
            {"ARI_score": ari_score, "AMI_score": ami_score, "y_true": y_true, "y_pred": y_pred},
            properties_file,
            indent=4
        )
