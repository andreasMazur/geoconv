from geoconv.tensorflow.layers.barycentric_coordinates import BarycentricCoordinates
from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.conv_geodesic import ConvGeodesic
from geoconv.tensorflow.layers.pooling.angular_max_pooling import AngularMaxPooling

import tensorflow as tf
import tensorflow_probability as tfp


class Covariance(tf.keras.layers.Layer):
    @tf.function
    def call(self, inputs):
        return tf.map_fn(tfp.stats.covariance, inputs)


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
                 dropout_rate=0.3):
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

        # Input normalization
        self.normalize = tf.keras.layers.Normalization(axis=-1, name="input_normalization")

        #################
        # EMBEDDING PART
        #################
        # Determine which layer type shall be used
        assert variant in ["dirac", "geodesic"], "Please choose a layer type from: ['dirac', 'geodesic']."
        self.layer_type = ConvGeodesic if variant == "geodesic" else ConvDirac

        # Define vertex embedding architecture
        self.isc_layers, self.bn_layers = [], []
        for dim in isc_layer_dims:
            self.isc_layers.append(
                self.layer_type(
                    amt_templates=dim,
                    template_radius=template_radius,
                    activation="elu",
                    name="ISC",
                    rotation_delta=rotation_delta,
                    initializer="glorot_uniform"
                )
            )
            self.bn_layers.append(tf.keras.layers.BatchNormalization(axis=-1, name="batch_normalization"))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.amp = AngularMaxPooling()

        ######################
        # CLASSIFICATION PART
        ######################
        # Define covariance layer
        self.cov = Covariance()

        # Define classification layer
        self.flatten = tf.keras.layers.Flatten()
        self.clf = tf.keras.layers.Dense(units=10 if modelnet10 else 40)

    def call(self, inputs, **kwargs):
        # Compute barycentric coordinates from 3D coordinates
        bc = self.bc_layer(inputs)

        # Normalize signal
        signal = self.normalize(inputs)

        # Compute vertex embeddings
        for idx in range(len(self.isc_layers)):
            signal = self.dropout(signal)
            signal = self.isc_layers[idx]([signal, bc])
            signal = self.amp(signal)
            signal = self.bn_layers[idx](signal)

        # Compute covariance matrix from vertex-embeddings
        signal = self.cov(signal)
        signal = self.flatten(signal)

        # Return classification logits
        return self.clf(signal)
