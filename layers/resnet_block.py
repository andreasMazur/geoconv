from tensorflow.keras import Model

from layers.angular_max_pooling import AngularMaxPooling
from layers.geodesic_conv import ConvGeodesic

import tensorflow as tf


class ResNetBlock(Model):

    def __init__(self,
                 amt_kernel,
                 activation="relu",
                 name=None):
        if name is not None:
            super().__init__(name=name)
        else:
            super().__init__()
        self.activation = activation
        self.amt_kernel = amt_kernel
        self.amp = AngularMaxPooling()
        self.first_call = True
        self.gc1 = None
        self.gc2 = None
        self.add = tf.keras.layers.Add()

    @tf.function
    def call(self, inputs):
        signal_input, b_coordinates = inputs
        if self.first_call:
            # dim(input) = dim(output) such that adding is possible
            self.gc1 = ConvGeodesic(signal_input.shape[2], self.amt_kernel, self.activation)
            self.gc2 = ConvGeodesic(signal_input.shape[2], self.amt_kernel, self.activation)
            self.first_call = False

        signal = self.gc1([signal_input, b_coordinates])
        signal = self.amp(signal)

        signal = self.gc2([signal, b_coordinates])
        signal = self.amp(signal)

        signal = self.add([signal, signal_input])
        return signal
