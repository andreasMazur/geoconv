from geoconv.tensorflow.backbone import ResNetBlock
from geoconv.tensorflow.layers import ConvDirac
from geoconv.tensorflow.layers import ConvGeodesic
from geoconv.tensorflow.layers import AngularMaxPooling

import tensorflow as tf


AMOUNT_VERTICES = 6890
SIG_DIM = 544


class FaustVertexClassifier(tf.keras.Model):
    def __init__(
        self,
        template_radius,
        isc_layer_dims,
        middle_layer_dim=1024,
        variant=None,
        normalize_input=True,
        rotation_delta=1,
        dropout_rate=0.3,
        output_rotation_delta=1,
        l1_reg=0.3,
        clf_output=True,
        signal_dim=SIG_DIM,
        initializer="glorot_uniform",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.normalize_input = normalize_input

        # Determine which layer type shall be used
        variant = "dirac" if variant is None else variant
        if variant not in ["dirac", "geodesic"]:
            raise RuntimeError(
                f"'{variant}' is not a valid network type. Please select a valid variant from ['dirac', 'geodesic']."
            )

        ##############
        # Down blocks
        ##############
        self.isc_layers_down = []
        self.batch_normalizations_down = []
        for idx in range(len(isc_layer_dims)):
            self.isc_layers_down.append(
                ResNetBlock(
                    amt_templates=isc_layer_dims[idx],
                    template_radius=template_radius,
                    rotation_delta=rotation_delta,
                    conv_type=variant,
                    activation="elu",
                    input_dim=signal_dim if idx == 0 else isc_layer_dims[idx - 1],
                    initializer=initializer,
                )
            )

        ###############
        # Middle block
        ###############
        self.isc_layers_middle = ResNetBlock(
            amt_templates=middle_layer_dim,
            template_radius=template_radius,
            rotation_delta=rotation_delta,
            conv_type=variant,
            activation="elu",
            input_dim=isc_layer_dims[-1],
            initializer=initializer,
        )

        ############
        # Up blocks
        ############
        self.isc_layers_up = []
        self.batch_normalizations_up = []

        isc_layer_dims = isc_layer_dims[::-1]
        for idx in range(len(isc_layer_dims)):
            self.isc_layers_up.append(
                ResNetBlock(
                    amt_templates=isc_layer_dims[idx],
                    template_radius=template_radius,
                    rotation_delta=rotation_delta,
                    conv_type=variant,
                    activation="elu",
                    input_dim=-1,
                    initializer=initializer,
                )
            )

        # Auxiliary layers
        if self.normalize_input:
            self.normalize = tf.keras.layers.Normalization(
                axis=-1, name="input_normalization"
            )
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        # Classification layer
        self.clf_output = clf_output
        if clf_output:
            clf_type = ConvDirac if variant == "dirac" else ConvGeodesic
            self.clf = clf_type(
                amt_templates=AMOUNT_VERTICES,
                template_radius=template_radius,
                activation="linear",
                name="output",
                rotation_delta=output_rotation_delta,
                template_regularizer=tf.keras.regularizers.L1(l1=l1_reg),
                bias_regularizer=None,
                initializer=initializer,
            )
            self.amp = AngularMaxPooling()
        else:
            self.clf = tf.keras.layers.Identity(name="output")

        # Concat layer
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, **kwargs):
        signal, bc = inputs
        if self.normalize_input:
            signal = self.normalize(signal)

        # Compute vertex embeddings (down-scaling)
        down_scaling = []
        for idx in range(len(self.isc_layers_down)):
            signal = self.dropout(signal)
            signal = self.isc_layers_down[idx]([signal, bc])
            down_scaling.append(signal)

        # Middle
        signal = self.isc_layers_middle([signal, bc])

        # Compute vertex embeddings (up-scaling)
        down_scaling = down_scaling[::-1]
        for idx in range(len(self.isc_layers_up)):
            signal = self.dropout(signal)
            signal = self.concat([signal, down_scaling[idx]])
            signal = self.isc_layers_up[idx]([signal, bc])

        # Output
        # signal = self.concat([signal, down_scaling[-1]])
        if self.clf_output:
            signal = self.clf([signal, bc])
            return self.amp(signal)
        else:
            return self.clf(signal)
