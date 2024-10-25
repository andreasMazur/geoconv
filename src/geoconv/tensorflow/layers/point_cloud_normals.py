from geoconv.tensorflow.utils.compute_shot_lrf import knn_shot_lrf

import tensorflow as tf
import tensorflow_probability as tfp


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
        covs = tfp.stats.covariance(tf.gather(normals, neighborhoods_indices), sample_axis=1)
        covs = tf.reshape(covs, (tf.shape(covs)[0], 9))
        return covs / tf.linalg.norm(covs, axis=-1, keepdims=True)
