from tensorflow.keras import Model

from geoconv.geodesic_conv import ConvGeodesic

import tensorflow as tf


class ResNetBlock(Model):

    def __init__(self,
                 kernel_size,
                 amt_kernel,
                 activation="relu",
                 name=None):
        if name is not None:
            super().__init__(name=name)
        else:
            super().__init__()
        self.activation = activation
        self.kernel_size = kernel_size
        self.amt_kernel = amt_kernel
        self.first_call = True
        self.gc1 = None
        self.gc2 = None
        self.add = tf.keras.layers.Add()

    @tf.function
    def call(self, inputs):
        signal_input, b_coordinates = inputs
        if self.first_call:
            # dim(input) = dim(output) such that adding is possible
            self.gc1 = ConvGeodesic(self.kernel_size, signal_input.shape[2], self.amt_kernel, self.activation)
            self.gc2 = ConvGeodesic(self.kernel_size, signal_input.shape[2], self.amt_kernel, self.activation)
            self.first_call = False
        signal = self.gc1([signal_input, b_coordinates])
        signal = self.gc2([signal, b_coordinates])
        signal = self.add([signal, signal_input])
        return signal

