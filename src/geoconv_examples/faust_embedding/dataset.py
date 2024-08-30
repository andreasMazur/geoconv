from geoconv.utils.data_generator import preprocessed_shape_generator
from geoconv_examples.faust.dataset import faust_generator

import tensorflow as tf
import numpy as np


def faust_embedding_generator(dataset_path,
                              n_radial,
                              n_angular,
                              template_radius,
                              is_train,
                              only_signal=False,
                              seed=42,
                              gen_info_file=""):
    # Load reference shape
    psg = preprocessed_shape_generator(
        dataset_path,
        filter_list=["SIGNAL", f"BC_{n_radial}_{n_angular}_{template_radius}"],
        shuffle_seed=None,
        split=[0]
    )
    for elements in psg:
        shot_ref = elements[0][0]
        bc_ref = elements[1][0]

    # Load other shapes
    fg = faust_generator(dataset_path, n_radial, n_angular, template_radius, is_train, only_signal, seed, gen_info_file)
    for (shot, bc), gt in fg:
        yield (shot_ref, bc_ref, shot, bc), gt


def load_faust_embedding_gen(path_to_zip,
                             n_radial,
                             n_angular,
                             template_radius,
                             is_train,
                             only_signal=False,
                             seed=42,
                             gen_info_file="",
                             batch_size=1):
    if only_signal:
        output_signature = tf.TensorSpec(shape=(None, 544), dtype=tf.float32)  # Signal
    else:
        output_signature = (
            (
                tf.TensorSpec(shape=(None, 544,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,) + (n_radial, n_angular) + (3, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 544,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,) + (n_radial, n_angular) + (3, 2), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )

    return tf.data.Dataset.from_generator(
        faust_embedding_generator,
        args=(
            path_to_zip,  # dataset_path
            n_radial,  # n_radial
            n_angular,  # n_angular
            np.array(template_radius, np.float64),  # template_radius
            is_train,  # is_train
            only_signal,  # only_signal
            seed,  # seed
            gen_info_file  # gen_info_file
        ),
        output_signature=output_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
