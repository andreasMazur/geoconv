from geoconv.layers.angular_max_pooling import AngularMaxPooling
from geoconv.layers.conv_dirac import ConvDirac

from tensorflow.python.client import device_lib

import tensorflow as tf


class Imcnn(tf.keras.Model):
    def __init__(self, template_radius, splits):
        super().__init__()
        self.amp = AngularMaxPooling()
        self.gpus = [device_name for device_name in device_lib.list_local_devices() if "GPU" in device_name.name]

        # with tf.device(self.gpus[0].name):
        self.conv1 = ConvDirac(
            amt_templates=96,
            template_radius=template_radius,
            activation="relu",
            name="ISC_layer_1",
            splits=splits
        )
        self.conv2 = ConvDirac(
            amt_templates=256,
            template_radius=template_radius,
            activation="relu",
            name="ISC_layer_2",
            splits=splits,
        )

        # with tf.device(self.gpus[1].name):
        self.conv3 = ConvDirac(
            amt_templates=384,
            template_radius=template_radius,
            activation="relu",
            name="ISC_layer_3",
            splits=splits,
        )

        # with tf.device(self.gpus[2].name):
        self.conv4 = ConvDirac(
            amt_templates=384,
            template_radius=template_radius,
            activation="relu",
            name="ISC_layer_4",
            splits=splits,
        )
        self.conv5 = ConvDirac(
            amt_templates=256,
            template_radius=template_radius,
            activation="relu",
            name="ISC_layer_5",
            splits=splits
        )
        self.output_layer = tf.keras.layers.Dense(6890)

    def call(self, inputs, training=None, mask=None):
        # with tf.device(self.gpus[0].name):
        signal, bc = inputs
        signal = self.conv1([signal, bc])
        signal = self.amp(signal)
        signal = self.conv2([signal, bc])
        signal = self.amp(signal)
        # with tf.device(self.gpus[1].name):
        signal = self.conv3([signal, bc])
        signal = self.amp(signal)
        # with tf.device(self.gpus[2].name):
        signal = self.conv4([signal, bc])
        signal = self.amp(signal)
        signal = self.conv5([signal, bc])
        signal = self.amp(signal)
        return self.output_layer(signal)
