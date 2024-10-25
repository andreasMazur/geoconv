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
    @tf.function(jit_compile=True)
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        return tf.reshape(tfp.stats.covariance(inputs, sample_axis=1), (input_shape[0], input_shape[-1] ** 2))


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
                 dropout_rate=0.3,
                 initializer="glorot_uniform",
                 pooling="cov",
                 return_vertex_embeddings=False):
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
        self.normals = PointCloudNormals(neighbors_for_lrf=neighbors_for_lrf)

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
                    activation="elu",
                    input_dim=-1 if idx == 0 else isc_layer_dims[idx - 1],
                    initializer=initializer
                )
            )

        ######################
        # CLASSIFICATION PART
        ######################
        assert pooling in ["cov", "max"], "Please set your pooling to either 'cov' or 'max'."
        if pooling == "cov":
            self.pool = Covariance()
        else:
            self.pool = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")

        # Define classification layer
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.clf = tf.keras.layers.Dense(units=10 if modelnet10 else 40)

        self.return_vertex_embeddings = return_vertex_embeddings

    def call(self, inputs, **kwargs):
        # Shift point-cloud centroid into 0
        coordinates = self.center(inputs)

        # Compute barycentric coordinates from 3D coordinates
        bc = self.bc_layer(coordinates)

        # Compute normals
        signal = self.normals(coordinates)

        # Compute vertex embeddings
        for idx in range(len(self.isc_layers)):
            signal = self.isc_layers[idx]([signal, bc])

        if self.return_vertex_embeddings:
            return signal

        # Covariance-pool
        signal = self.pool(signal)

        # Return classification logits
        signal = self.dropout(signal)
        return self.clf(signal)
