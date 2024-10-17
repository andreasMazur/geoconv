from geoconv.tensorflow.layers.barycentric_coordinates import BarycentricCoordinates
from geoconv.tensorflow.layers.tangent_projections import TangentProjections
from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.conv_geodesic import ConvGeodesic
from geoconv.tensorflow.layers.pooling.angular_max_pooling import AngularMaxPooling

import tensorflow as tf
import tensorflow_probability as tfp


class Covariance(tf.keras.layers.Layer):
    @tf.function
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
                 use_covariance=True):
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

        # Tangent projections
        self.projection_layer = TangentProjections(n_neighbors=n_neighbors)

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
        self.pool = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")

        # Define classification layer
        self.clf = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, activation="elu"),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Dense(256, activation="elu"),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Dense(units=10 if modelnet10 else 40),
        ])

        self.use_covariance = use_covariance
        self.cov = Covariance()

    def coordinates_to_input(self, coordinates):
        # Project into tangent planes
        projection = self.projection_layer(coordinates)

        # Return covariance matrix
        if self.use_covariance:
            return self.cov(projection)
        else:
            return projection

    def call(self, inputs, **kwargs):
        # Compute barycentric coordinates from 3D coordinates
        bc = self.bc_layer(inputs)

        # Get input data
        signal = self.coordinates_to_input(inputs)

        # Normalize signal
        signal = self.normalize(signal)

        # Compute vertex embeddings
        for idx in range(len(self.isc_layers)):
            signal = self.dropout(signal)
            signal = self.isc_layers[idx]([signal, bc])
            signal = self.amp(signal)
            signal = self.bn_layers[idx](signal)

        # Global max-pool
        signal = self.pool(signal)

        # Return classification logits
        return self.clf(signal)
