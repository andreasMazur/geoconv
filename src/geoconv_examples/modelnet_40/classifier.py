from geoconv.tensorflow.backbone.resnet_block import ResNetBlock
from geoconv.tensorflow.layers.point_cloud_normals import PointCloudNormals
from geoconv.tensorflow.layers.barycentric_coordinates import BarycentricCoordinates

import tensorflow as tf
import tensorflow_probability as tfp


class ShiftPointCloud(tf.keras.layers.Layer):
    @tf.function(jit_compile=True)
    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs, axis=1, keepdims=True)


class Covariance(tf.keras.layers.Layer):
    @tf.function
    def call(self, inputs):
        feature_dim = tf.shape(inputs)[-1]
        cov = tfp.stats.covariance(inputs, sample_axis=1)
        lower_tri_mask = tf.linalg.band_part(tf.ones(shape=(feature_dim, feature_dim)), 0, -1)
        return tf.map_fn(lambda c: tf.boolean_mask(c, lower_tri_mask), cov)


class ModelNetClf(tf.keras.Model):
    def __init__(self,
                 n_radial,
                 n_angular,
                 template_radius,
                 isc_layer_dims,
                 neighbors_for_lrf=16,
                 modelnet10=False,
                 variant=None,
                 rotation_delta=1,
                 initializer="glorot_uniform",
                 pooling="cov",
                 noise_stddev=1e-3):
        super().__init__()

        #############
        # INPUT PART
        #############
        # Init barycentric coordinates layer
        self.bc_layer = BarycentricCoordinates(
            n_radial=n_radial, n_angular=n_angular, neighbors_for_lrf=neighbors_for_lrf
        )
        self.bc_layer.adapt(template_radius=template_radius)

        # For centering point clouds
        self.center = ShiftPointCloud()

        #################
        # EMBEDDING PART
        #################
        # Determine which layer type shall be used
        assert variant in ["dirac", "geodesic"], "Please choose a layer type from: ['dirac', 'geodesic']."

        # Define vertex embedding architecture
        self.isc_layers = []
        for idx, dim in enumerate(isc_layer_dims):
            self.isc_layers.append(
                ResNetBlock(
                    amt_templates=dim,
                    template_radius=template_radius,
                    rotation_delta=rotation_delta,
                    conv_type=variant,
                    activation="relu",
                    input_dim=-1 if idx == 0 else isc_layer_dims[idx - 1],
                    initializer=initializer
                )
            )

        ######################
        # CLASSIFICATION PART
        ######################
        assert pooling in ["cov", "max", "avg"], "Please set your pooling to either 'cov', 'max' or 'avg'."
        if pooling == "cov":
            self.pool = Covariance()
        elif pooling == "avg":
            self.pool = tf.keras.layers.GlobalAvgPool1D(data_format="channels_last")
        else:
            self.pool = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")

        # Define classification layer
        self.output_dim = 10 if modelnet10 else 40
        self.clf = tf.keras.layers.Dense(units=self.output_dim, activation="linear")

        # Add noise during training
        self.noise = tf.keras.layers.GaussianNoise(stddev=noise_stddev)

    def call(self, inputs, training=False, **kwargs):
        # Shift point-cloud centroid into 0
        coordinates = self.center(inputs)

        # Compute barycentric coordinates from 3D coordinates
        bc = self.bc_layer(coordinates)

        # Add noise
        signal = self.noise(coordinates, training=training)

        # Compute vertex embeddings
        for idx in range(len(self.isc_layers)):
            signal = self.isc_layers[idx]([signal, bc], training=training)

        # Get normalized point-cloud embeddings
        signal = self.pool(signal)

        return self.clf(signal)
