from geoconv.utils.data_generator import preprocessed_shape_generator

import tensorflow as tf
import numpy as np


def faust_generator(dataset_path, n_radial, n_angular, template_radius, is_train, only_signal=False, seed=42):
    split_indices = list(range(80)) if is_train else list(range(80, 100))

    # Load barycentric coordinates
    psg = preprocessed_shape_generator(
        dataset_path,
        filter_list=["SIGNAL", f"BC_{n_radial}_{n_angular}_{template_radius}"],
        shuffle_seed=seed,
        split=split_indices
    )

    # Set seed for permutations
    np.random.seed(seed)

    for elements in psg:
        shot = elements[0][0]
        bc = elements[1][0]

        assert bc.shape[0] == shot.shape[0], "Numbers of shot-descriptors and barycentric coordinates do not match!"

        # Compute permutation (assumed to be unknown for the model)
        permutation = np.arange(shot.shape[0]).astype(np.int32)
        np.random.shuffle(permutation)

        # Compute ground truth (inverse permutation)
        inverse_permutation = np.zeros(shot.shape[0]).astype(np.int32)
        for idx, perm_idx in enumerate(permutation):
            inverse_permutation[perm_idx] = int(idx)

        # Permute shot descriptors
        # It holds that:
        #   shot[permutation[x]] = shot_perm[x] <=> shot[x] = shot_perm[inverse_permutation[x]]  -- (I)
        shot_perm = shot[permutation]

        # Permute barycentric coordinates:
        bc_perm = np.array(bc)
        # 1.) Change stored BC-vertex indices such that new ones point to the same SHOT-descriptors as before.
        # Substitute 'x' in (I) with bc[k, r, a, i, 0]:
        #   shot[bc[k, r, a, i, 0]] == shot_perm[inverse_permutation[bc[k, r, a, i, 0]]]
        # Store 'inverse_permutation[bc[k, r, a, i, 0]]' in bc_perm[k, r, a, i, 0]:
        #   bc_perm[k, r, a, i, 0] = inverse_permutation[bc[k, r, a, i, 0]]
        # Now we have:
        #   shot[bc[k, r, a, i, 0]] = shot_perm[bc_perm[k, r, a, i, 0]]
        bc_perm[:, :, :, :, 0] = inverse_permutation[bc_perm[:, :, :, :, 0].astype(np.int32)]

        # 2.) Permute order of center-vertices
        #   bc[k, r, a, i, 0] = bc_perm[inverse_permutation[k], r, a, i, 0]
        #   <=> bc[permutation[k], r, a, i, 0] = bc_perm[k, r, a, i, 0]
        bc_perm = bc_perm[permutation]

        # Combining both we thus have in total:
        #   shot[bc[k, r, a, i, 0]] = shot_perm[bc_perm[inverse_permutation[k], r, a, i, 0]]
        # assert (
        #         shot[bc[:, :, :, :, 0].astype(np.int32)] \
        #           == shot_perm[bc_perm[inverse_permutation, :, :, :, 0].astype(np.int32)]
        # ).all()

        if only_signal:
            yield shot_perm
        else:
            yield (shot_perm, bc_perm), permutation


def load_preprocessed_faust(path_to_zip,
                            n_radial,
                            n_angular,
                            template_radius,
                            is_train,
                            only_signal=False):
    if only_signal:
        output_signature = tf.TensorSpec(shape=(None, 544,), dtype=tf.float32)  # Signal  (3D coordinates)
    else:
        output_signature = (
            (
                tf.TensorSpec(shape=(None, 544,), dtype=tf.float32),  # Signal  (3D coordinates)
                tf.TensorSpec(shape=(None,) + (n_radial, n_angular) + (3, 2), dtype=tf.float32)  # Barycentric Coordinates
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )

    return tf.data.Dataset.from_generator(
        faust_generator,
        args=(
            path_to_zip, n_radial, n_angular, np.array(template_radius, np.float64), is_train, only_signal
        ),
        output_signature=output_signature
    ).batch(1).prefetch(tf.data.AUTOTUNE)
