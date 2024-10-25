from geoconv.tensorflow.utils.compute_shot_lrf import knn_shot_lrf

import tensorflow as tf


class PointCloudNormals(tf.keras.layers.Layer):
    def __init__(self, neighbors_for_lrf=32):
        super().__init__()
        self.neighbors_for_lrf = neighbors_for_lrf

    @tf.function(jit_compile=True)
    def call(self, vertices):
        return tf.map_fn(self.call_helper, vertices)

    @tf.function(jit_compile=True)
    def call_helper(self, vertices):
        lrfs, _, neighborhoods_indices = knn_shot_lrf(self.neighbors_for_lrf, vertices)
        return tf.reshape(lrfs, (tf.shape(lrfs)[0], 9))  # lrfs[:, :, 0]
