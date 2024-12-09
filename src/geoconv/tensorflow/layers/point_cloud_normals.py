from geoconv.tensorflow.utils.compute_shot_lrf import knn_shot_lrf

import tensorflow as tf


class PointCloudNormals(tf.keras.layers.Layer):
    def __init__(self, neighbors_for_lrf=16):
        super().__init__()
        self.neighbors_for_lrf = neighbors_for_lrf

    @tf.function(jit_compile=True)
    def call(self, vertices):
        return tf.map_fn(self.call_helper, vertices)

    @tf.function(jit_compile=True)
    def call_helper(self, vertices):
        lrfs, _, neighborhoods_indices = knn_shot_lrf(self.neighbors_for_lrf, vertices)
        normals = lrfs[:, :, 0]
        normals = tf.gather(normals, neighborhoods_indices)
        covs = tf.einsum("nvi,nvj->nij", normals, normals)

        lower_tri_mask = tf.linalg.band_part(tf.ones(shape=(3, 3)), 0, -1)
        covs = tf.map_fn(lambda c: tf.boolean_mask(c, lower_tri_mask), covs)

        return covs / tf.linalg.norm(covs, axis=-1, keepdims=True)
