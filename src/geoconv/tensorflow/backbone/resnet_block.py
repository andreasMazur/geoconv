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
                 input_dim=-1,
                 initializer="glorot_uniform",
                 template_regularizer=None,
                 bias_regularizer=None):
        super(ResNetBlock, self).__init__()

        assert conv_type in ["dirac", "geodesic"], "Please choose a layer type from: ['dirac', 'geodesic']."
        self.layer_type = ConvGeodesic if conv_type == "geodesic" else ConvDirac

        # block 1
        self.conv1 = self.layer_type(
            amt_templates=amt_templates,
            template_radius=template_radius,
            activation="linear",
            name="ResNetBlock_1",
            rotation_delta=rotation_delta,
            initializer=initializer,
            template_regularizer=template_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, name=f"batch_normalization")
        self.amp1 = AngularMaxPooling()

        # block 2
        self.conv2 = self.layer_type(
            amt_templates=amt_templates,
            template_radius=template_radius,
            activation="linear",
            name="ResNetBlock_2",
            rotation_delta=rotation_delta,
            initializer=initializer,
            template_regularizer=template_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1, name=f"batch_normalization")
        self.amp2 = AngularMaxPooling()

        self.add = tf.keras.layers.Add()

        self.rotation_delta = rotation_delta

        self.rescale = input_dim != amt_templates
        if self.rescale:
            self.conv_rescale = self.layer_type(
                amt_templates=amt_templates,
                template_radius=template_radius,
                activation="linear",
                name="ResNetBlock_rescale",
                rotation_delta=rotation_delta,
                initializer=initializer,
                template_regularizer=template_regularizer,
                bias_regularizer=bias_regularizer
            )
            self.bn_rescale = tf.keras.layers.BatchNormalization(axis=-1, name=f"batch_normalization")
            self.amp_rescale = AngularMaxPooling()
        self.output_activation = tf.keras.activations.get(activation)

    @tf.function
    def call(self, inputs, training=False, **kwargs):
        input_signal, bc = inputs

        if tf.constant(training):
            orientations = tf.range(start=0, limit=tf.shape(bc)[3], delta=self.rotation_delta)
        else:
            orientations = tf.range(tf.shape(bc)[3])

        # F(x)
        signal = self.conv1([input_signal, bc], orientations)
        signal = self.amp1(signal)
        signal = self.bn1(signal)
        signal = self.output_activation(signal)

        signal = self.conv2([signal, bc], orientations)
        signal = self.amp2(signal)
        signal = self.bn2(signal)

        if self.rescale:
            # W x
            input_signal = self.conv_rescale([input_signal, bc], orientations)
            input_signal = self.amp_rescale(input_signal)
            input_signal = self.bn_rescale(input_signal)

        # F(x) + W x
        signal = self.add([signal, input_signal])

        return self.output_activation(signal)
