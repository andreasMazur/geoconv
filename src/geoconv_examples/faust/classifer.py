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
                 rotation_delta=1,
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
        if variant == "dirac":
            self.layer_type = ConvDirac
        else:
            self.layer_type = ConvGeodesic

        ##############
        # Down blocks
        ##############
        self.isc_layers_down = []
        self.batch_normalizations_down = []
        for idx in range(len(isc_layer_dims) - 1):
            self.isc_layers_down.append(
                self.layer_type(
                    amt_templates=isc_layer_dims[idx],
                    template_radius=template_radius,
                    activation="elu",
                    name=f"ISC_layer_down_{idx}",
                    rotation_delta=rotation_delta
                )
            )
            self.batch_normalizations_down.append(
                tf.keras.layers.BatchNormalization(axis=-1, name=f"BN_layer_down_{idx}")
            )

        ###############
        # Middle block
        ###############
        self.isc_layers_middle = ConvDirac(
            amt_templates=isc_layer_dims[-1],
            template_radius=template_radius,
            activation="elu",
            name=f"ISC_layer_middle",
            rotation_delta=1
        )
        self.batch_normalizations_middle = tf.keras.layers.BatchNormalization(
            axis=-1, name=f"BN_layer_middle"
        )

        ############
        # Up blocks
        ############
        self.isc_layers_up = []
        self.batch_normalizations_up = []

        isc_layer_dims = isc_layer_dims[1::-1]
        for idx in range(len(isc_layer_dims)):
            self.isc_layers_up.append(
                ConvDirac(
                    amt_templates=isc_layer_dims[idx],
                    template_radius=template_radius,
                    activation="elu",
                    name=f"ISC_layer_up_{idx}",
                    rotation_delta=1
                )
            )
            self.batch_normalizations_up.append(tf.keras.layers.BatchNormalization(axis=-1, name=f"BN_layer_up_{idx}"))

        # Auxiliary layers
        self.amp = AngularMaxPooling()
        if self.normalize_input:
            self.normalize = tf.keras.layers.Normalization(axis=-1, name="input_normalization")
        self.dropout = tf.keras.layers.Dropout(rate=0.2)

        # Classification layer
        self.output_dense = tf.keras.layers.Dense(AMOUNT_VERTICES, name="output")

        # Concat layer
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, **kwargs):
        signal, bc = inputs
        if self.normalize_input:
            signal = self.normalize(signal)
        signal = self.dropout(signal)

        # Compute vertex embeddings (down-scaling)
        down_scaling = []
        for idx in range(len(self.isc_layers_down)):
            signal = self.isc_layers_down[idx]([signal, bc])
            signal = self.amp(signal)
            signal = self.batch_normalizations_down[idx](signal)
            down_scaling.append(signal)

        # Middle
        signal = self.isc_layers_middle([signal, bc])
        signal = self.amp(signal)
        signal = self.batch_normalizations_middle(signal)

        # Compute vertex embeddings (up-scaling)
        down_scaling = down_scaling[::-1]
        for idx in range(len(self.isc_layers_up)):
            signal = self.isc_layers_up[idx]([signal, bc])
            signal = self.amp(signal)
            signal = self.batch_normalizations_up[idx](signal)
            signal = self.concat([signal, down_scaling[idx]])

        # Output
        return self.output_dense(signal)
