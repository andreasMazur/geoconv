from tensorflow.keras import Model

from geoconv.geodesic_conv import ConvGeodesic

import tensorflow as tf


class ResNetBlock(Model):

    def __init__(self,
                 kernel_size,
                 amt_kernel,
                 output_dim,
                 activation="relu",
                 kernel_size_2=None,
                 amt_kernel_2=None,
                 activation_2=None,
                 name=""):
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.amt_kernel = amt_kernel
        self.kernel_size_2 = kernel_size_2
        self.amt_kernel_2 = amt_kernel_2

        self.gc1 = ConvGeodesic(kernel_size, output_dim, amt_kernel, activation)
        if kernel_size_2 is None or amt_kernel_2 is None or activation_2 is None:
            self.gc2 = ConvGeodesic(kernel_size, output_dim, amt_kernel, activation)
        else:
            self.gc2 = ConvGeodesic(kernel_size_2, output_dim, amt_kernel_2, activation_2)

    @tf.function
    def call(self, inputs):
        signal_input, b_coordinates = inputs
        signal = self.gc1([signal_input, b_coordinates])
        signal = self.gc2([signal, b_coordinates])
        return signal + signal_input
