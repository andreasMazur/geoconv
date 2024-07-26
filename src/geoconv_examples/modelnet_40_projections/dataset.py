from geoconv.utils.data_generator import preprocessed_shape_generator

import tensorflow as tf
import numpy as np


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

MODELNET10_CLASSES = {
    "bathtub": 0,
    "bed": 1,
    "chair": 2,
    "desk": 3,
    "dresser": 4,
    "monitor": 5,
    "night_stand": 6,
    "sofa": 7,
    "table": 8,
    "toilet": 9
}


def modelnet_generator(dataset_path, is_train, only_signal=False, modelnet10=False, gen_info_file=""):
    prefix = "train" if is_train else "test"
    if modelnet10:
        filter_list = list(MODELNET10_CLASSES.keys())
        filter_list = [f"{prefix}/{c}_.*/vertices" for c in filter_list]
    else:
        filter_list = [f"{prefix}.*vertices"]

    # Load sampled vertices from preprocessed dataset
    psg = preprocessed_shape_generator(
        dataset_path, filter_list=filter_list, shuffle_seed=None, filter_gpc_systems=False, gen_info_file=gen_info_file
    )

    for [(vertices, vertices_path)] in psg:
        if only_signal:
            yield vertices
        else:
            if modelnet10:
                yield vertices, np.array(MODELNET10_CLASSES[vertices_path.split("/")[1]]).reshape(1)
            else:
                yield vertices, np.array(MODELNET_CLASSES[vertices_path.split("/")[1]]).reshape(1)


def load_preprocessed_modelnet(dataset_path,
                               is_train,
                               batch_size=4,
                               only_signal=False,
                               modelnet10=False,
                               gen_info_file=""):
    if only_signal:
        output_signature = tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
    else:
        output_signature = (
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    return tf.data.Dataset.from_generator(
        modelnet_generator,
        args=(dataset_path, is_train, only_signal, modelnet10, gen_info_file),
        output_signature=output_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
