from geoconv.utils.data_generator import preprocessed_shape_generator

import numpy as np
import tensorflow as tf


def faust_generator(dataset_path,
                    temp_conf_1,
                    temp_radius_1,
                    temp_conf_2,
                    temp_radius_2,
                    temp_conf_3,
                    temp_radius_3,
                    is_train,
                    only_signal=False,
                    seed=42,
                    gen_info_file=""):
    split_indices = list(range(80)) if is_train else list(range(80, 100))

    # Load barycentric coordinates
    n_radial_1, n_angular_1 = temp_conf_1
    n_radial_2, n_angular_2 = temp_conf_2
    n_radial_3, n_angular_3 = temp_conf_3
    psg = preprocessed_shape_generator(
        dataset_path,
        filter_list=[
            "SIGNAL",
            f"BC_{n_radial_1}_{n_angular_1}_{temp_radius_1}",
            f"BC_{n_radial_2}_{n_angular_2}_{temp_radius_2}",
            f"BC_{n_radial_3}_{n_angular_3}_{temp_radius_3}"
        ],
        shuffle_seed=int(seed),
        split=split_indices,
        gen_info_file=gen_info_file
    )

    # Set seed for permutations
    np.random.seed(seed)

    for elements in psg:
        shot = elements[0][0]
        bc_1 = elements[1][0]
        bc_2 = elements[2][0]
        bc_3 = elements[3][0]

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

        def perm_bc(bc):
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
            return bc_perm
        bc_perm_1 = perm_bc(bc_1)
        bc_perm_2 = perm_bc(bc_2)
        bc_perm_3 = perm_bc(bc_3)

        if only_signal:
            yield shot_perm
        else:
            yield (shot_perm, bc_perm_1, bc_perm_2, bc_perm_3), permutation


def load_preprocessed_faust(path_to_zip,
                            temp_conf_1,
                            temp_conf_2,
                            temp_conf_3,
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
                tf.TensorSpec(shape=(None, 544,), dtype=tf.float32),  # Signal
                tf.TensorSpec(shape=(None,) + (temp_conf_1[0], temp_conf_1[1]) + (3, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(None,) + (temp_conf_2[0], temp_conf_2[1]) + (3, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(None,) + (temp_conf_3[0], temp_conf_3[1]) + (3, 2), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )

    return tf.data.Dataset.from_generator(
        faust_generator,
        args=(
            path_to_zip,  # dataset_path
            temp_conf_1[:2],
            np.array(temp_conf_1[2], np.float64),
            temp_conf_2[:2],
            np.array(temp_conf_2[2], np.float64),
            temp_conf_3[:2],
            np.array(temp_conf_3[2], np.float64),
            is_train,  # is_train
            only_signal,  # only_signal
            seed,  # seed
            gen_info_file  # gen_info_file
        ),
        output_signature=output_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
