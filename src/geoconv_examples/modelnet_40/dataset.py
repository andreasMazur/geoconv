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
    "airplane": 0.9595767964523114,
    "bathtub": 1.0255222391562435,
    "bed": 0.9640413295452341,
    "bench": 1.000229381555837,
    "bookshelf": 0.9615814493229973,
    "bottle": 0.9757382380376598,
    "bowl": 1.0608019520992245,
    "car": 0.994657405403399,
    "chair": 0.9526476279715312,
    "cone": 1.0018173874548533,
    "cup": 1.0447165325409995,
    "curtain": 1.0109910875228898,
    "desk": 0.9940359430296744,
    "door": 1.0238735732134756,
    "dresser": 0.9940359430296744,
    "flower_pot": 1.0071782931321742,
    "glass_box": 1.0007489082457406,
    "guitar": 1.005281819618632,
    "keyboard": 1.0085114382614726,
    "lamp": 1.0166076444100496,
    "laptop": 1.0071782931321742,
    "mantel": 0.9810271397945649,
    "monitor": 0.9665738149594261,
    "night_stand": 0.9940359430296744,
    "person": 1.0372647117667582,
    "piano": 0.9883738707793455,
    "plant": 0.9869487146179379,
    "radio": 1.0266636347487377,
    "range_hood": 1.0207852065645695,
    "sink": 1.0149031438797569,
    "sofa": 0.9578222902742033,
    "stairs": 1.0166076444100496,
    "stool": 1.0357743932927013,
    "table": 0.9711391567439316,
    "tent": 1.002927187851981,
    "toilet": 0.9749338383703365,
    "tv_stand": 0.9831332853261725,
    "vase": 0.9660344605452533,
    "wardrobe": 1.038030612184975,
    "xbox": 1.027247667723378
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
    "bathtub": 1.0460434265452505,
    "bed": 0.9845625169342411,
    "chair": 0.9731688153605382,
    "desk": 1.0145571304186813,
    "dresser": 1.0145571304186813,
    "monitor": 0.9870950023484331,
    "night_stand": 1.0145571304186813,
    "sofa": 0.9783434776632103,
    "table": 0.9916603441329386,
    "toilet": 0.9954550257593435
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
