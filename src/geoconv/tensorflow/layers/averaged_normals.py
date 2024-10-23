from geoconv.tensorflow.utils.compute_shot_lrf import knn_shot_lrf

import tensorflow as tf


class AveragedNormals(tf.keras.layers.Layer):
    def __init__(self, neighbors_for_lrf=128, neighbors_for_avg=128):
        super().__init__()
        self.neighbors_for_lrf = neighbors_for_lrf
        self.neighbors_for_avg = neighbors_for_avg

    @tf.function(jit_compile=True)
    def call(self, vertices):
        return tf.map_fn(self.call_helper, vertices)

    @tf.function(jit_compile=True)
    def call_helper(self, vertices):
        lrfs, _, neighborhoods_indices = knn_shot_lrf(self.neighbors_for_lrf, vertices)
        if self.neighbors_for_avg > 1:
            avg_normal = tf.reduce_mean(
                tf.gather(lrfs[:, 0, :], neighborhoods_indices[:, :self.neighbors_for_avg]), axis=1
            )
            return avg_normal / tf.expand_dims(tf.linalg.norm(avg_normal, axis=-1), axis=-1)
        else:
            return lrfs[:, 0, :]
