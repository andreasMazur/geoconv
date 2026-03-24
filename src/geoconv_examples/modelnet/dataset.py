from tqdm import tqdm

import numpy as np
import tensorflow as tf
import os
import math


MAX_N_VERTICES = 6042


def adapt_generator(layer, path, set_type, chart_max_radius, method, do_zero_pad):
    gen = generator(path, set_type, chart_max_radius, method, return_rotations=False, do_zero_pad=do_zero_pad)
    for (vertices, bc), _ in tqdm(gen, postfix="Adapting normalization layer..."):
        yield layer(vertices[None, ...])


def get_class_name_and_number(filepath):
    filepath = filepath.split("/")[-2]
    if filepath.count("_") == 2:
        cls_1, cls_2, number = filepath.split("_")
        cls = f"{cls_1}_{cls_2}"
    else:
        cls, number = filepath.split("_")
    return cls, int(number)


def get_class_counts(zip_content):
    class_counts = {}
    for filepath in zip_content:
        cls, _ = get_class_name_and_number(filepath)
        if cls not in class_counts.keys():
            class_counts[cls] = 1
        else:
            class_counts[cls] += 1
    return class_counts


def zero_pad(array, max_nodes=MAX_N_VERTICES):
    zeros = np.zeros((max_nodes - array.shape[0], *array.shape[1:]))
    return np.concatenate([array, zeros], axis=0)


def generator(path,
              set_type,
              chart_max_radius,
              method,
              return_rotations=True,
              random_seed=42,
              do_zero_pad=True):
    """Returns a 'generator'-object for the ModelNet dataset.

    Parameters
    ----------
    path: str
        The path to the preprocessed zip-file of ModelNet.
    set_type: str
        The set type. Either: 'train', 'val', 'test' or 'all'.
    chart_max_radius: float
        The upper bound radius for the local charts.
    method: str
        The used preprocessing method.
    return_rotations: bool
        Whether to return the rotation angles for the parallel transport.
    random_seed: int
        The random seed used to shuffle the data.
    do_zero_pad: bool
        Whether to zero pad all arrays expect the ground truth label to the same size in the first axis as the
        largest array in the dataset.

    Returns
    -------
    generator:
        A ModelNet generator.
    """
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    if isinstance(set_type, bytes):
        set_type = set_type.decode("utf-8")
    if isinstance(method, bytes):
        method = method.decode("utf-8")

    zip_file = np.load(path, allow_pickle=True)

    # Load zip content
    zip_content = [
        f for f in zip_file.files
        if f"mn10_{method}_{'_'.join(f'{chart_max_radius}'.split('.'))}" in f and "barycentric_coordinates" in f
    ]
    zip_content.sort(key=get_class_name_and_number)

    # Get desired set type
    if set_type == "train":
        # Filter down to training data
        zip_content_temp = [f for f in zip_content if "train" in f]

        # Get the class counts
        class_counts = get_class_counts(zip_content_temp)

        # Filter down to first 90% of training data per class
        zip_content = []
        for cls, count in class_counts.items():
            cls_zip_content = [f for f in zip_content_temp if cls in f][:math.floor(count * 0.9)]
            zip_content.extend(cls_zip_content)
    elif set_type == "val":
        # Filter down to training data
        zip_content_temp = [f for f in zip_content if "train" in f]

        # Get the class counts
        class_counts = get_class_counts(zip_content_temp)

        # Filter down to last 10% of training data per class
        zip_content = []
        for cls, count in class_counts.items():
            cls_zip_content = [f for f in zip_content_temp if cls in f][math.floor(count * 0.9):]
            zip_content.extend(cls_zip_content)
    elif set_type == "test":
        zip_content = [f for f in zip_content if "test" in f]
    elif set_type == "all":
        pass
    else:
        raise RuntimeError(f"Invalid set_type: '{set_type}'. Select either 'train', 'validation', 'test' or 'all'.")

    # Shuffle the data
    np.random.seed(random_seed)
    np.random.shuffle(zip_content)

    for filepath in zip_content:
        # Load barycentric coordinates
        barycentric_coordinates = zip_file[filepath]

        # TODO: Find what's causing the NaN-angles.
        if np.isnan(barycentric_coordinates).any():
            continue

        # Load related shape info
        file_dir = os.path.dirname(filepath)
        vertices = zip_file[f"{file_dir}/vertices"]
        gt = zip_file[f"{file_dir}/ground_truth"]

        # Zero pad vertices and bc to a common shape
        mask = np.zeros((MAX_N_VERTICES,)).astype(np.bool_)
        mask[range(vertices.shape[0])] = True
        if do_zero_pad:
            vertices = zero_pad(vertices)
            barycentric_coordinates = zero_pad(barycentric_coordinates)

        # Yield dataset elements
        if return_rotations:
            yield (vertices, barycentric_coordinates, mask), gt
        else:
            yield (vertices, barycentric_coordinates[..., :2], mask), gt


def dataset(path,
            set_type,
            n_radial,
            n_angular,
            chart_max_radius,
            method,
            return_rotations=True,
            random_seed=42,
            do_zero_pad=True):
    """Returns a 'tensorflow dataset'-object for the ModelNet dataset.

    Parameters
    ----------
    path: str
        The path to the preprocessed zip-file of ModelNet.
    set_type: str
        The set type. Either: 'train', 'test' or 'all'.
    n_radial: int
        The number of radial coordinates of the template.
    n_angular: int
        The number of angular coordinates of the template.
    chart_max_radius: tf.Tensor
        The upper bound radius for the local charts.
    method: str
        The used preprocessing method.
    return_rotations: bool
        Whether to return the rotation angles for the parallel transport.
    random_seed: int
        The random seed used to shuffle the data.
    do_zero_pad: bool
        Whether to zero pad all arrays expect the ground truth label to the same size in the first axis as the
        largest array in the dataset.

    Returns
    -------
    tf.data.Dataset:
        A ModelNet dataset.
    """
    if return_rotations:
        bc_shape = (3, 3)
    else:
        bc_shape = (3, 2)
    n_vertices = 6042 if do_zero_pad else None

    output_signature = (
        (
            tf.TensorSpec(shape=(n_vertices, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(n_vertices,) + (n_radial, n_angular) + bc_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(n_vertices,), dtype=tf.bool),
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )

    return tf.data.Dataset.from_generator(
        generator,
        args=(path, set_type, chart_max_radius, method, return_rotations, random_seed, do_zero_pad),
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE).batch(1)
