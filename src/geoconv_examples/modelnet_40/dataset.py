from geoconv.utils.data_generator import preprocessed_shape_generator

from io import BytesIO

import numpy as np
import trimesh
import tensorflow as tf


MODELNET40_TOTAL = 7917
MODELNET40_FOLDS = {
    -1: list(range(0, MODELNET40_TOTAL)),
    0: list(range(0, 1583)),
    1: list(range(1583, 1583 * 2)),
    2: list(range(1583 * 2, 1583 * 3)),
    3: list(range(1583 * 3, 1583 * 4)),
    4: list(range(1583 * 4, MODELNET40_TOTAL))
}
MODELNET40_TRAIN_SPLITS = {
    0: MODELNET40_FOLDS[1] + MODELNET40_FOLDS[2] + MODELNET40_FOLDS[3] + MODELNET40_FOLDS[4],
    1: MODELNET40_FOLDS[0] + MODELNET40_FOLDS[2] + MODELNET40_FOLDS[3] + MODELNET40_FOLDS[4],
    2: MODELNET40_FOLDS[1] + MODELNET40_FOLDS[0] + MODELNET40_FOLDS[3] + MODELNET40_FOLDS[4],
    3: MODELNET40_FOLDS[1] + MODELNET40_FOLDS[2] + MODELNET40_FOLDS[0] + MODELNET40_FOLDS[4],
    4: MODELNET40_FOLDS[1] + MODELNET40_FOLDS[2] + MODELNET40_FOLDS[3] + MODELNET40_FOLDS[0],
}
MODELNET_CLASSES = {
    "airplane": 0,
    "bathtub": 1,
    "bed": 2,
    "bench": 3,
    "bookshelf": 4,
    "bottle": 5,
    "bowl": 6,
    "car": 7,
    "chair": 8,
    "cone": 9,
    "cup": 10,
    "curtain": 11,
    "desk": 12,
    "door": 13,
    "dresser": 14,
    "flower_pot": 15,
    "glass_box": 16,
    "guitar": 17,
    "keyboard": 18,
    "lamp": 19,
    "laptop": 20,
    "mantel": 21,
    "monitor": 22,
    "night_stand": 23,
    "person": 24,
    "piano": 25,
    "plant": 26,
    "radio": 27,
    "range_hood": 28,
    "sink": 29,
    "sofa": 39,
    "stairs": 31,
    "stool": 32,
    "table": 33,
    "tent": 34,
    "toilet": 35,
    "tv_stand": 36,
    "vase": 37,
    "wardrobe": 38,
    "xbox": 39
}


def modelnet_generator(path_to_zip, n_radial, n_angular, template_radius, is_train, split):
    # Choose train or test split
    if is_train:
        split = MODELNET40_TRAIN_SPLITS[split]
    else:
        split = MODELNET40_FOLDS[split]

    # Load barycentric coordinates
    psg = preprocessed_shape_generator(
        path_to_zip,
        filter_list=["stl", f"BC_{n_radial}_{n_angular}_{template_radius}"],
        shuffle_seed=42 if split != -1 else None,
        split=split
    )

    for elements in psg:
        vertices = trimesh.load_mesh(BytesIO(elements[0][0]), file_type="stl").vertices
        bc = elements[1][0]

        assert bc.shape[0] == vertices.shape[0], "Numbers of vertices and barycentric coordinates do not match!"

        yield (vertices, bc), np.array(MODELNET_CLASSES[elements[0][1].split("/")[1]]).reshape(1)


def load_preprocessed_modelnet(path_to_zip, n_radial, n_angular, template_radius, is_train, split):
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 3,), dtype=tf.float32),  # Signal  (3D coordinates)
            tf.TensorSpec(shape=(None,) + (n_radial, n_angular) + (3, 2), dtype=tf.float32)  # Barycentric Coordinates
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )

    return tf.data.Dataset.from_generator(
        modelnet_generator,
        args=(path_to_zip, n_radial, n_angular, np.array(template_radius, np.float64), is_train, split),
        output_signature=output_signature
    ).prefetch(1)
