from geoconv.tensorflow.utils.compute_shot_decr import shot_descr
from geoconv.tensorflow.utils.compute_shot_lrf import knn_shot_lrf

import tensorflow as tf


class PointCloudShotDescriptor(tf.keras.layers.Layer):
    def __init__(
        self,
        neighbors_for_lrf=16,
        azimuth_bins=8,
        elevation_bins=2,
        radial_bins=2,
        histogram_bins=11,
        sphere_radius=0.0,
    ):
        super().__init__()
        self.neighbors_for_lrf = neighbors_for_lrf
        self.azimuth_bins = azimuth_bins
        self.elevation_bins = elevation_bins
        self.radial_bins = radial_bins
        self.histogram_bins = histogram_bins
        self.sphere_radius = sphere_radius

    @tf.function(jit_compile=True)
    def call(self, vertices):
        return tf.map_fn(self.call_helper, vertices)

    @tf.function(jit_compile=True)
    def call_helper(self, vertices):
        lrfs, neighborhoods, neighborhoods_indices = knn_shot_lrf(
            self.neighbors_for_lrf, vertices
        )
        if self.sphere_radius > 0.0:
            return shot_descr(
                neighborhoods=neighborhoods,
                normals=lrfs[..., 0],
                neighborhood_indices=neighborhoods_indices,
                radius=self.sphere_radius,
                azimuth_bins=self.azimuth_bins,
                elevation_bins=self.elevation_bins,
                radial_bins=self.radial_bins,
                histogram_bins=self.histogram_bins,
            )
        else:
            return shot_descr(
                neighborhoods=neighborhoods,
                normals=lrfs[..., 0],
                neighborhood_indices=neighborhoods_indices,
                radius=tf.reduce_max(tf.linalg.norm(neighborhoods, axis=-1)),
                azimuth_bins=self.azimuth_bins,
                elevation_bins=self.elevation_bins,
                radial_bins=self.radial_bins,
                histogram_bins=self.histogram_bins,
            )
