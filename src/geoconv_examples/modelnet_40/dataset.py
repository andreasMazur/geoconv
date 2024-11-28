from geoconv.utils.data_generator import preprocessed_shape_generator

import tensorflow as tf
import numpy as np
import random


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
    "sofa": 30,
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
MN_CLASS_WEIGHTS = {
    "airplane": 0.39309105431309904,
    "bathtub": 2.3214622641509433,
    "bed": 0.4778155339805825,
    "bench": 1.4223988439306359,
    "bookshelf": 0.43020104895104894,
    "bottle": 0.7345522388059702,
    "bowl": 3.844921875,
    "car": 1.2491116751269036,
    "chair": 0.2767997750281215,
    "cone": 1.4735029940119762,
    "cup": 3.114873417721519,
    "curtain": 1.7831521739130434,
    "desk": 1.230375,
    "door": 2.2575688073394495,
    "dresser": 1.230375,
    "flower_pot": 1.6515100671140939,
    "glass_box": 1.4390350877192983,
    "guitar": 1.5875806451612904,
    "keyboard": 1.6970689655172413,
    "lamp": 1.9844758064516128,
    "laptop": 1.6515100671140939,
    "mantel": 0.8664612676056338,
    "monitor": 0.5291935483870968,
    "night_stand": 1.230375,
    "person": 2.7963068181818183,
    "piano": 1.0652597402597404,
    "plant": 1.0253125,
    "radio": 2.3661057692307694,
    "range_hood": 2.139782608695652,
    "sink": 1.9224609375,
    "sofa": 0.361875,
    "stairs": 1.9844758064516128,
    "stool": 2.734166666666667,
    "table": 0.6277423469387755,
    "tent": 1.5096625766871166,
    "toilet": 0.7153343023255814,
    "tv_stand": 0.9216292134831461,
    "vase": 0.5180526315789473,
    "wardrobe": 2.828448275862069,
    "xbox": 2.3890776699029126
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
MN10_CLASS_WEIGHTS = {
    "bathtub": 3.7650943396226415,
    "bed": 0.7749514563106796,
    "chair": 0.44893138357705287,
    "desk": 1.9955,
    "dresser": 1.9955,
    "monitor": 0.8582795698924731,
    "night_stand": 1.9955,
    "sofa": 0.5869117647058824,
    "table": 1.0181122448979592,
    "toilet": 1.1601744186046512
}

DEBUG_DATASET_SIZE = 100


def shuffle_directive(shape_dict):
    shape_dict_keys = list(shape_dict.keys())
    random.shuffle(shape_dict_keys)
    return {key: shape_dict[key] for key in shape_dict_keys}


def modelnet_generator(dataset_path,
                       set_type,
                       modelnet10=False,
                       gen_info_file="",
                       debug_data=False,
                       in_one_hot=False):
    if isinstance(set_type, bytes):
        set_type = set_type.decode("utf-8")

    if set_type not in ["train", "test", "all"]:
        raise RuntimeError(f"Unknown dataset type: '{set_type}' Please select from: ['train', 'test', 'all'].")

    set_type = "" if set_type == "all" else set_type
    if modelnet10:
        filter_list = list(MODELNET10_CLASSES.keys())
        filter_list = [f"{set_type}/{c}_.*/vertices" for c in filter_list]
    else:
        filter_list = [f"{set_type}.*vertices"]

    # Load sampled vertices from preprocessed dataset
    psg = preprocessed_shape_generator(
        zipfile_path=dataset_path,
        filter_list=filter_list,
        batch_size=1,
        generator_info=gen_info_file,
        directive=shuffle_directive
    )

    for idx, shape in enumerate(psg):
        point_cloud, file_path = shape[0]

        # Check whether this dataset is intended for debugging
        if idx == DEBUG_DATASET_SIZE and debug_data:
            break

        # Check whether to use ModelNet10 labels
        if modelnet10:
            n_classes = 10
            label = np.array(MODELNET10_CLASSES[file_path.split("/")[1]]).reshape(1)
        else:
            n_classes = 40
            label = np.array(MODELNET_CLASSES[file_path.split("/")[1]]).reshape(1)

        # Check whether labels are supposed to be one-hot encoded
        if in_one_hot:
            label = np.eye(n_classes)[label[0]]

        yield point_cloud, label


def load_preprocessed_modelnet(dataset_path,
                               set_type,
                               batch_size=4,
                               modelnet10=False,
                               gen_info_file="",
                               debug_data=False,
                               in_one_hot=False):
    n_classes = 10 if modelnet10 else 40
    if in_one_hot:
        output_signature = (
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(n_classes,), dtype=tf.float32)
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )

    return tf.data.Dataset.from_generator(
        modelnet_generator,
        args=(dataset_path, set_type, modelnet10, gen_info_file, debug_data, in_one_hot),
        output_signature=output_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
