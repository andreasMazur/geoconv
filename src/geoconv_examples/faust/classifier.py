from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.conv_geodesic import ConvGeodesic
from geoconv.tensorflow.layers.pooling.angular_max_pooling import AngularMaxPooling

import tensorflow as tf


AMOUNT_VERTICES = 6890


class FaustVertexClassifier(tf.keras.Model):
    def __init__(self,
                 template_radius,
                 isc_layer_dims=None,
                 variant=None,
                 normalize_input=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_input = normalize_input

        # Determine which layer type shall be used
        variant = "dirac" if variant is None else variant
        if variant not in ["dirac", "geodesic"]:
            raise RuntimeError(
                f"'{variant}' is not a valid network type. Please select a valid variant from ['dirac', 'geodesic']."
            )

        # Initialize ISC blocks
        self.isc_layers = []
        self.batch_normalizations = []
        for idx in range(len(isc_layer_dims)):
            if variant == "dirac":
                self.isc_layers.append(
                    ConvDirac(
                        amt_templates=isc_layer_dims[idx],
                        template_radius=template_radius,
                        activation="relu",
                        name=f"ISC_layer_{idx}",
                        rotation_delta=1
                    )
                )
            else:
                self.isc_layers.append(
                    ConvGeodesic(
                        amt_templates=isc_layer_dims[idx],
                        template_radius=template_radius,
                        activation="relu",
                        name=f"ISC_layer_{idx}",
                        rotation_delta=1
                    )
                )
            self.batch_normalizations.append(tf.keras.layers.BatchNormalization(axis=-1, name=f"BN_layer_{idx}"))

        # Auxiliary layers
        self.amp = AngularMaxPooling()
        if self.normalize_input:
            self.normalize = tf.keras.layers.Normalization(axis=-1, name="input_normalization")
        self.dropout = tf.keras.layers.Dropout(rate=0.3)

        # Classification layer
        self.output_dense = tf.keras.layers.Dense(AMOUNT_VERTICES, name="output")

    def call(self, inputs, **kwargs):
        signal, bc = inputs
        if self.normalize_input:
            signal = self.normalize(signal)
        signal = self.dropout(signal)

        # Compute vertex embeddings
        for idx in range(len(self.isc_layers)):
            signal = self.isc_layers[idx]([signal, bc])
            signal = self.amp(signal)
            signal = self.batch_normalizations[idx](signal)

        # Output
        return self.output_dense(signal)
