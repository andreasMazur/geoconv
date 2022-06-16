import tensorflow as tf


class MoNetConv(tf.keras.layers.Layer):

    def __init__(self):
        """Remember initialization parameter"""
        super(MoNetConv, self).__init__()

    def build(self):
        """Create layer weights"""
        pass

    def call(self, inputs):
        """The computation rule of that layer

        :param inputs: SHOT-descriptors of surface points [NOT CONTINUOUS MANIFOLD]
        """
        pass
