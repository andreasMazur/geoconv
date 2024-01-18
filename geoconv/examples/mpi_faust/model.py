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

        #################
        # Handling Input
        #################
        self.normalize = keras.layers.Normalization(axis=-1)
        self.downsize_dense = keras.layers.Dense(64, activation="relu", name="downsize")
        self.downsize_bn = keras.layers.BatchNormalization(axis=-1, name="BN_downsize")

        ##################
        # Global Features
        ##################
        self.output_dims = [20, 20, 20, 20, 50]
        self.isc_layers = []
        self.bn_layers = []
        self.do_layers = []
        self.amp_layers = []
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
            self.do_layers.append(keras.layers.Dropout(rate=0.2, name=f"DO_layer_{idx}"))
            if idx < len(self.output_dims) - 1:
                self.amp_layers.append(AngularMaxPooling())

        ##################
        # Local Features
        ##################
        self.local_isc = ConvDirac(
            amt_templates=50,
            template_radius=self.template_radius,
            activation="relu",
            name=f"local_ISC_layer",
            splits=self.splits,
            rotation_delta=self.rotation_delta
        )

        #################
        # Global + Local
        #################
        self.concat_layer = keras.layers.Concatenate(axis=1)
        self.concat_amp = AngularMaxPooling()

        #########
        # Output
        #########
        self.output_dense = keras.layers.Dense(6890, name="output")

    def call(self, inputs, orientations=None, training=None, mask=None):
        #################
        # Handling Input
        #################
        signal, bc = inputs
        signal = self.normalize(signal)
        signal = self.downsize_dense(signal)
        signal = self.downsize_bn(signal)

        ##################
        # Global Features
        ##################
        global_signal = self.isc_layers[0]([signal, bc])
        global_signal = self.amp_layers[0](global_signal)
        global_signal = self.bn_layers[0](global_signal)
        global_signal = self.do_layers[0](global_signal)
        for idx in range(1, len(self.output_dims)):
            global_signal = self.isc_layers[idx]([global_signal, bc])
            if idx < len(self.output_dims) - 1:
                global_signal = self.amp_layers[idx](global_signal)
                global_signal = self.bn_layers[idx](global_signal)
                global_signal = self.do_layers[idx](global_signal)

        ##################
        # Local Features
        ##################
        local_signal = self.local_isc([signal, bc])

        #################
        # Global + Local
        #################
        combined_signal = self.concat_layer([global_signal, local_signal])
        combined_signal = self.concat_amp(combined_signal)

        return self.output_dense(combined_signal)
