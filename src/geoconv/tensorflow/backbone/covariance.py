import tensorflow as tf
import tensorflow_probability as tfp


class Covariance(tf.keras.layers.Layer):

    @tf.function
    def call(self, inputs):
        feature_dim = tf.shape(inputs)[-1]
        cov = tfp.stats.covariance(inputs, sample_axis=1)
        lower_tri_mask = tf.linalg.band_part(tf.ones(shape=(feature_dim, feature_dim)), 0, -1)
        return tf.map_fn(lambda c: tf.boolean_mask(c, lower_tri_mask), cov)
