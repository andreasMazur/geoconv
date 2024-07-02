from geoconv.utils.data_generator import preprocessed_shape_generator

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


def modelnet_generator(dataset_path, is_train):
    if is_train:
        filter_list = ["train.*vertices"]
    else:
        filter_list = ["test.*vertices"]

    # Load sampled vertices from preprocessed dataset
    psg = preprocessed_shape_generator(dataset_path, filter_list=filter_list, shuffle_seed=42, filter_gpc_systems=False)

    for [(vertices, vertices_path)] in psg:
        yield vertices
