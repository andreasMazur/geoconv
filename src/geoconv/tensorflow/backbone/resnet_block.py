from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.conv_geodesic import ConvGeodesic
from geoconv.tensorflow.layers.pooling.angular_max_pooling import AngularMaxPooling

import tensorflow as tf


class ResNetBlock(tf.keras.Model):
    def __init__(self,
                 amt_templates,
                 template_radius,
                 rotation_delta,
                 conv_type="dirac",
                 activation="relu",
                 input_dim=-1):
        super(ResNetBlock, self).__init__()

        assert conv_type in ["dirac", "geodesic"], "Please choose a layer type from: ['dirac', 'geodesic']."
        self.layer_type = ConvGeodesic if conv_type == "geodesic" else ConvDirac

        self.conv1 = self.layer_type(
            amt_templates=amt_templates,
            template_radius=template_radius,
            activation=activation,
            name="ResNetBlock_1",
            rotation_delta=rotation_delta
        )
        self.conv2 = self.layer_type(
            amt_templates=amt_templates,
            template_radius=template_radius,
            activation="linear",
            name="ResNetBlock_2",
            rotation_delta=rotation_delta
        )
        self.amp = AngularMaxPooling()
        self.add = tf.keras.layers.Add()

        self.rescale = input_dim != amt_templates
        if self.rescale:
            self.conv_rescale = self.layer_type(
                amt_templates=amt_templates,
                template_radius=template_radius,
                activation="linear",
                name="ResNetBlock_rescale",
                rotation_delta=rotation_delta
            )
        self.output_activation = tf.keras.activations.get(activation)

    def call(self, inputs, **kwargs):
        input_signal, bc = inputs

        # F(x)
        signal = self.conv1([input_signal, bc])
        signal = self.amp(signal)
        signal = self.conv2([signal, bc])
        signal = self.amp(signal)

        if self.rescale:
            # W x
            input_signal = self.conv_rescale([input_signal, bc])
            input_signal = self.amp(input_signal)

        # F(x) + W x
        signal = self.add([signal, input_signal])

        return self.output_activation(signal)
