from geoconv.utils.data_generator import preprocessed_shape_generator

from io import BytesIO

import trimesh


MODELNET40_TOTAL = 7917
MODELNET40_SPLITS = {
    "fold_1": list(range(0, 1583)),
    "fold_2": list(range(1598, 1583 * 2)),
    "fold_3": list(range(1583 * 2, 1583 * 3)),
    "fold_4": list(range(1583 * 3, 1583 * 4)),
    "fold_5": list(range(1583 * 4, MODELNET40_TOTAL))
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


def modelnet_generator(bc_path, n_radial, n_angular, template_radius, split):
    # Load barycentric coordinates
    psg = preprocessed_shape_generator(
        bc_path, filter_list=["stl", f"BC_{n_radial}_{n_angular}_{template_radius}"], shuffle_seed=42, split=split
    )

    for elements in psg:
        bc = elements[0][0]
        vertices = trimesh.load_mesh(BytesIO(elements[1][0]), file_type="stl").vertices

        assert bc.shape[0] == vertices.shape[0], "Number of vertices and barycentric coordinates does not match!"

        yield (vertices, bc), MODELNET_CLASSES[elements[0][1].split("/")[1]]
