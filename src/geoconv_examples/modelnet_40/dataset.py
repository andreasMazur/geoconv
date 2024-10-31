from geoconv.utils.data_generator import preprocessed_shape_generator

import tensorflow as tf
import numpy as np
import random

# Normalized inverse frequencies, only considering training split!
CLASS_WEIGHTS = tf.constant([
    0.10223642172523961,
    0.6037735849056604,
    0.12427184466019417,
    0.3699421965317919,
    0.11188811188811187,
    0.191044776119403,
    1.0,
    0.32487309644670054,
    0.07199100112485939,
    0.38323353293413176,
    0.810126582278481,
    0.46376811594202894,
    0.32,
    0.5871559633027523,
    0.32,
    0.42953020134228187,
    0.37426900584795325,
    0.4129032258064516,
    0.4413793103448276,
    0.5161290322580645,
    0.42953020134228187,
    0.22535211267605634,
    0.13763440860215054,
    0.32,
    0.7272727272727273,
    0.27705627705627706,
    0.26666666666666666,
    0.6153846153846154,
    0.5565217391304348,
    0.5,
    0.09411764705882353,
    0.5161290322580645,
    0.711111111111111,
    0.16326530612244897,
    0.39263803680981596,
    0.18604651162790697,
    0.2397003745318352,
    0.13473684210526315,
    0.735632183908046,
    0.6213592233009708
])


CLASS_WEIGHTS_MN10 = tf.constant([
    1.0,
    0.2058252427184466,
    0.11923509561304836,
    0.5299999999999999,
    0.5299999999999999,
    0.2279569892473118,
    0.5299999999999999,
    0.15588235294117647,
    0.2704081632653061,
    0.3081395348837209
])


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

DEBUG_DATASET_SIZE = 100


def triplet_directive(shape_dict):
    classes = list(set([f[0].split("/")[1] for f in list(shape_dict.values())]))
    class_elements = [[f for f in list(shape_dict.values()) if c in f[0]] for c in classes]

    shape_dict_values = list(shape_dict.values())
    random.shuffle(shape_dict_values)

    new_shape_dict = {}
    for file in shape_dict_values:
        file_class = classes.index(file[0].split("/")[1])
        anchor = random.choice(class_elements[file_class])

        random_other_cls = random.choice([cls for cls in range(len(classes)) if cls != file_class])
        negative = random.choice(class_elements[random_other_cls])

        new_shape_dict[f"{'/'.join(file[0].split('/')[:-1])}_anchor"] = anchor
        new_shape_dict[f"{'/'.join(file[0].split('/')[:-1])}_positive"] = file
        new_shape_dict[f"{'/'.join(file[0].split('/')[:-1])}_negative"] = negative

    return new_shape_dict


def modelnet_generator(dataset_path,
                       set_type,
                       modelnet10=False,
                       gen_info_file="",
                       debug_data=False):
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
        batch_size=3,
        generator_info=gen_info_file,
        directive=triplet_directive
    )

    for idx, triplet in enumerate(psg):
        if idx == DEBUG_DATASET_SIZE and debug_data:
            break
        anchor, positive, negative = triplet[0][0], triplet[1][0], triplet[2][0]
        if modelnet10:
            positive_class = np.array(MODELNET10_CLASSES[triplet[1][1].split("/")[1]]).reshape(1)
        else:
            positive_class = np.array(MODELNET_CLASSES[triplet[1][1].split("/")[1]]).reshape(1)
        yield tf.stack([anchor, positive, negative], axis=-2), positive_class


def load_preprocessed_modelnet(dataset_path,
                               set_type,
                               batch_size=4,
                               modelnet10=False,
                               gen_info_file="",
                               debug_data=False):
    return tf.data.Dataset.from_generator(
        modelnet_generator,
        args=(dataset_path, set_type, modelnet10, gen_info_file, debug_data),
        output_signature=(
            tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
