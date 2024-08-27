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
                 rotation_delta=1):
        super().__init__()

        # Init barycentric coordinates layer
        self.bc_layer = BarycentricCoordinates(
            n_radial=n_radial,
            n_angular=n_angular,
            n_neighbors=n_neighbors,
            template_scale=None
        )
        self.bc_layer.adapt(template_radius=template_radius)

        # Determine which layer type shall be used
        variant = "dirac" if variant is None else variant
        if variant not in ["dirac", "geodesic"]:
            raise RuntimeError(
                f"'{variant}' is not a valid network type. Please select a valid variant from ['dirac', 'geodesic']."
            )

        # Init ISC block
        self.isc_layers = []
        for idx in range(len(isc_layer_dims)):
            if variant == "dirac":
                self.isc_layers.append(
                    ConvDirac(
                        amt_templates=isc_layer_dims[idx],
                        template_radius=template_radius,
                        activation="relu",
                        name=f"ISC_layer_{idx}",
                        rotation_delta=rotation_delta
                    )
                )
            else:
                self.isc_layers.append(
                    ConvGeodesic(
                        amt_templates=isc_layer_dims[idx],
                        template_radius=template_radius,
                        activation="relu",
                        name=f"ISC_layer_{idx}",
                        rotation_delta=rotation_delta
                    )
                )
        self.amp = AngularMaxPooling()
        self.cov = Covariance()

        # Define classification layer
        self.flatten = tf.keras.layers.Flatten()
        self.clf = tf.keras.layers.Dense(units=10 if modelnet10 else 40)

    def call(self, inputs, **kwargs):
        embedding = inputs

        # Compute barycentric coordinates
        bc = self.bc_layer(inputs)

        # Compute vertex embeddings
        for idx in range(len(self.isc_layers)):
            embedding = self.isc_layers[idx]([embedding, bc])
            embedding = self.amp(embedding)

        # Compute covariance matrix from vertex-embeddings
        embedding = self.cov(embedding)
        embedding = self.flatten(embedding)

        # Return classification logits
        return self.clf(embedding)
