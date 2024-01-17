from geoconv.layers.angular_max_pooling import AngularMaxPooling
from geoconv.layers.conv_dirac import ConvDirac
from tensorflow import keras

import tensorflow as tf


class Imcnn(tf.keras.Model):
    def __init__(self, signal_dim, kernel_size, template_radius, splits, rotation_delta):
        super().__init__()
        self.signal_dim = signal_dim
        self.kernel_size = kernel_size
        self.template_radius = template_radius
        self.splits = splits
        self.rotation_delta = rotation_delta
        self.output_dims = [96, 256, 384, 384, 256]

        self.amp = AngularMaxPooling()
        self.downsize_dense = keras.layers.Dense(64, activation="relu", name="downsize")
        self.downsize_bn = keras.layers.BatchNormalization(axis=-1, name="BN_downsize")

        self.isc_layers = []
        self.bn_layers = []
        self.do_layers = []
        for idx in range(len(self.output_dims)):
            self.isc_layers.append(
                ConvDirac(
                    amt_templates=self.output_dims[idx],
                    template_radius=self.template_radius,
                    activation="relu",
                    name=f"ISC_layer_{idx}",
                    splits=self.splits,
                    rotation_delta=self.rotation_delta
                )
            )
            self.bn_layers.append(keras.layers.BatchNormalization(axis=-1, name=f"BN_layer_{idx}"))
            self.do_layers.append(keras.layers.Dropout(rate=0.2))
        self.output_dense = keras.layers.Dense(6890, name="output")

    def call(self, inputs, orientations=None, training=None, mask=None):
        signal, bc = inputs
        signal = self.downsize_dense(signal)
        signal = self.downsize_bn(signal)
        for idx in range(len(self.output_dims)):
            signal = self.isc_layers[idx]([signal, bc])
            signal = self.amp(signal)
            signal = self.bn_layers[idx](signal)
            signal = self.do_layers[idx](signal)
        return self.output_dense(signal)
