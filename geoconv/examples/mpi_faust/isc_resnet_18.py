from geoconv.layers.angular_max_pooling import AngularMaxPooling
from geoconv.layers.conv_dirac import ConvDirac
from geoconv.models.resnet_block import ISCResidual

import tensorflow as tf


class ISCResnet18(tf.keras.Model):
    def __init__(self, splits, template_radius, rotation_deltas=None):
        super().__init__()

        if rotation_deltas is None:
            rotation_deltas = [1, 1, 1, 1]
        assert len(rotation_deltas) == 4, "You have to provide 4 rotation delta values."

        # Input normalization
        self.normalize = tf.keras.layers.Normalization(axis=-1, name="input_normalization")

        ############
        #  2x 64   #
        ############
        self.block_64_1 = ISCResidual(
            first_isc=ConvDirac,
            second_isc=ConvDirac,
            pool=AngularMaxPooling,
            layer_conf=[(64, rotation_deltas[0]), (64, rotation_deltas[0])],
            splits=splits,
            activation="relu",
            template_radius=template_radius,
            fit_dim=True
        )
        self.block_64_2 = ISCResidual(
            first_isc=ConvDirac,
            second_isc=ConvDirac,
            pool=AngularMaxPooling,
            layer_conf=[(64, rotation_deltas[0]), (64, rotation_deltas[0])],
            splits=splits,
            activation="relu",
            template_radius=template_radius
        )

        ############
        #  2x 128  #
        ############
        self.block_128_1 = ISCResidual(
            first_isc=ConvDirac,
            second_isc=ConvDirac,
            pool=AngularMaxPooling,
            layer_conf=[(128, rotation_deltas[1]), (128, rotation_deltas[1])],
            splits=splits,
            activation="relu",
            template_radius=template_radius,
            fit_dim=True
        )
        self.block_128_2 = ISCResidual(
            first_isc=ConvDirac,
            second_isc=ConvDirac,
            pool=AngularMaxPooling,
            layer_conf=[(128, rotation_deltas[1]), (128, rotation_deltas[1])],
            splits=splits,
            activation="relu",
            template_radius=template_radius
        )

        ############
        #  2x 256  #
        ############
        self.block_256_1 = ISCResidual(
            first_isc=ConvDirac,
            second_isc=ConvDirac,
            pool=AngularMaxPooling,
            layer_conf=[(256, rotation_deltas[2]), (256, rotation_deltas[2])],
            splits=splits,
            activation="relu",
            template_radius=,
            fit_dim=True
        )
        self.block_256_2 = ISCResidual(
            first_isc=ConvDirac,
            second_isc=ConvDirac,
            pool=AngularMaxPooling,
            layer_conf=[(256, rotation_deltas[2]), (256, rotation_deltas[2])],
            splits=splits,
            activation="relu",
            template_radius=template_radius
        )

        ############
        #  2x 512  #
        ############
        self.block_512_1 = ISCResidual(
            first_isc=ConvDirac,
            second_isc=ConvDirac,
            pool=AngularMaxPooling,
            layer_conf=[(512, rotation_deltas[3]), (512, rotation_deltas[3])],
            splits=splits,
            activation="relu",
            template_radius=template_radius,
            fit_dim=True
        )
        self.block_512_2 = ISCResidual(
            first_isc=ConvDirac,
            second_isc=ConvDirac,
            pool=AngularMaxPooling,
            layer_conf=[(512, rotation_deltas[3]), (512, rotation_deltas[3])],
            splits=splits,
            activation="relu",
            template_radius=template_radius
        )
        self.output_dense = tf.keras.layers.Dense(6890, name="output")

        self.forward_pass_list = [
            self.block_64_1,
            self.block_64_1,
            self.block_128_1,
            self.block_128_2,
            self.block_256_1,
            self.block_256_2,
            self.block_512_1,
            self.block_512_2
        ]

    def call(self, inputs, training=None, mask=None):
        #################
        # Handling Input
        #################
        signal, bc = inputs
        signal = self.normalize(signal)

        ###############
        # Forward pass
        ###############
        for layer in self.forward_pass_list[:-1]:
            signal = layer([signal, bc])
        signal = self.output_dense(signal)

        #########
        # Output
        #########
        return self.output_dense(signal)
