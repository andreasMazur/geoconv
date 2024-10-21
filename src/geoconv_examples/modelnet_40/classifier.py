from geoconv.tensorflow.backbone.resnet_block import ResNetBlock
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
        def compute_cov(x):
            x = tfp.stats.covariance(x, sample_axis=1)
            x_shape = tf.shape(x)
            return tf.reshape(x, [x_shape[0], x_shape[1] * x_shape[2]])
        return tf.map_fn(compute_cov, inputs)


class ModelNetClf(tf.keras.Model):
    def __init__(self,
                 n_neighbors,
                 n_radial,
                 n_angular,
                 template_radius,
                 isc_layer_dims,
                 modelnet10=False,
                 variant=None,
                 rotation_delta=1,
                 dropout_rate=0.3,
                 initializer="glorot_uniform"):
        super().__init__()

        #############
        # INPUT PART
        #############
        # Init barycentric coordinates layer
        self.bc_layer = BarycentricCoordinates(
            n_radial=n_radial,
            n_angular=n_angular,
            n_neighbors=n_neighbors,
            template_scale=None
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
                    activation="elu",
                    input_dim=-1 if idx == 0 else isc_layer_dims[idx - 1],
                    initializer=initializer
                )
            )
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        ######################
        # CLASSIFICATION PART
        ######################
        self.pool = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")

        # Define classification layer
        self.clf = tf.keras.layers.Dense(units=10 if modelnet10 else 40)

    def call(self, inputs, **kwargs):
        # Compute barycentric coordinates from 3D coordinates
        bc = self.bc_layer(inputs)

        # Shift point-cloud centroid into 0
        # signal = self.center(inputs)
        signal = inputs

        # Compute vertex embeddings
        for idx in range(len(self.isc_layers)):
            signal = self.dropout(signal)
            signal = self.isc_layers[idx]([signal, bc])

        # Global max-pool
        signal = self.pool(signal)

        # Return classification logits
        return self.clf(signal)
