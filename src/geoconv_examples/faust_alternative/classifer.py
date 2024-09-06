from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.conv_geodesic import ConvGeodesic
from geoconv.tensorflow.layers.pooling.angular_max_pooling import AngularMaxPooling
from geoconv_examples.faust.classifer import FaustVertexClassifier, AMOUNT_VERTICES

import tensorflow as tf


class MultiBranchClf(tf.keras.Model):
    def __init__(self,
                 temp_radius_1,
                 temp_radius_2,
                 temp_radius_3,
                 isc_layer_dims,
                 middle_layer_dim=1024,
                 variant=None,
                 normalize_input=True,
                 rotation_delta=1,
                 dropout_rate=0.3,
                 output_rotation_delta=1,
                 l1_reg=0.3,
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

        self.branch_1 = FaustVertexClassifier(
            temp_radius_1,
            isc_layer_dims=isc_layer_dims,
            middle_layer_dim=middle_layer_dim,
            variant=variant,
            normalize_input=True,
            rotation_delta=rotation_delta,
            dropout_rate=dropout_rate,
            output_rotation_delta=output_rotation_delta,
            l1_reg=l1_reg,
            clf_output=False
        )
        self.branch_2 = FaustVertexClassifier(
            temp_radius_2,
            isc_layer_dims=isc_layer_dims,
            middle_layer_dim=middle_layer_dim,
            variant=variant,
            normalize_input=True,
            rotation_delta=rotation_delta,
            dropout_rate=dropout_rate,
            output_rotation_delta=output_rotation_delta,
            l1_reg=l1_reg,
            clf_output=False
        )
        self.branch_3 = FaustVertexClassifier(
            temp_radius_3,
            isc_layer_dims=isc_layer_dims,
            middle_layer_dim=middle_layer_dim,
            variant=variant,
            normalize_input=True,
            rotation_delta=rotation_delta,
            dropout_rate=dropout_rate,
            output_rotation_delta=output_rotation_delta,
            l1_reg=l1_reg,
            clf_output=False
        )

        self.concat = tf.keras.layers.Concatenate()

        # Classification layer
        clf_type = ConvDirac if variant == "dirac" else ConvGeodesic
        self.clf = clf_type(
            amt_templates=AMOUNT_VERTICES,
            template_radius=temp_radius_2,
            activation="linear",
            name="output",
            rotation_delta=output_rotation_delta,
            template_regularizer=tf.keras.regularizers.L1(l1=l1_reg),
            bias_regularizer=None
        )
        self.amp = AngularMaxPooling()

    def call(self, inputs, **kwargs):
        signal, bc_1, bc_2, bc_3 = inputs

        # signal: [b, n, d]
        signal_1 = self.branch_1([signal, bc_1])
        signal_2 = self.branch_2([signal, bc_2])
        signal_3 = self.branch_3([signal, bc_3])

        # Concatenate branch results
        signal = self.concat([signal_1, signal_2, signal_3])

        # Output
        signal = self.clf([signal, bc_2])
        return self.amp(signal)
