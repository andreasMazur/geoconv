from geoconv.tensorflow.layers.barycentric_coordinates import BarycentricCoordinates
from geoconv_examples.faust.classifer import FaustVertexClassifier

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

        # Define vertex embedding architecture
        self.embedder = FaustVertexClassifier(
            template_radius,
            isc_layer_dims=isc_layer_dims,
            middle_layer_dim=64,
            variant=variant,
            normalize_input=True,
            rotation_delta=rotation_delta,
            dropout_rate=dropout_rate,
            l1_reg=0.0,
            initializer="glorot_uniform",
            clf_output=False
        )

        # Define covariance layer
        self.cov = Covariance()

        # Define classification layer
        self.flatten = tf.keras.layers.Flatten()
        self.clf = tf.keras.layers.Dense(units=10 if modelnet10 else 40)

    def call(self, inputs, **kwargs):
        # Compute barycentric coordinates
        bc = self.bc_layer(inputs)

        # Compute vertex embeddings
        embedding = self.embedder([inputs, bc])

        # Compute covariance matrix from vertex-embeddings
        embedding = self.cov(embedding)
        embedding = self.flatten(embedding)

        # Return classification logits
        return self.clf(embedding)
