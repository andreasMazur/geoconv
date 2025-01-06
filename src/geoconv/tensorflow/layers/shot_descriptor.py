from geoconv.tensorflow.utils.compute_shot_decr import shot_descr
from geoconv.tensorflow.utils.compute_shot_lrf import knn_shot_lrf

import tensorflow as tf


class PointCloudShotDescriptor(tf.keras.layers.Layer):
    def __init__(self, neighbors_for_lrf=16):
        super().__init__()
        self.neighbors_for_lrf = neighbors_for_lrf

    @tf.function(jit_compile=True)
    def call(self, vertices):
        return tf.map_fn(self.call_helper, vertices)

    @tf.function(jit_compile=True)
    def call_helper(self, vertices):
        lrfs, neighborhoods, neighborhoods_indices = knn_shot_lrf(self.neighbors_for_lrf, vertices)
        return shot_descr(
            neighborhoods=neighborhoods,
            normals=lrfs[:, :, 0],
            neighborhood_indices=neighborhoods_indices,
            radius=tf.reduce_max(neighborhoods),
            azimuth_bins=8,
            elevation_bins=2,
            radial_bins=2,
            histogram_bins=11
        )
