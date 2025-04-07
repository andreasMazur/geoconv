from geoconv.tensorflow.backbone.resnet_block import ResNetBlock
from geoconv.tensorflow.layers import ConvDirac
from geoconv.tensorflow.layers import ConvGeodesic
from geoconv.tensorflow.layers import AngularMaxPooling

import tensorflow as tf


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self,
                 intermediate_dims,
                 pre_bottleneck_dim,
                 template_radius,
                 rotation_delta,
                 variant,
                 initializer,
                 activation="relu"):
        super().__init__()

        # Remember parameters for 'init_conv'
        self.intermediate_convs = []
        self.intermediate_dims = intermediate_dims
        self.pre_bottleneck_conv = None
        self.pre_bottleneck_dim = pre_bottleneck_dim
        self.template_radius = template_radius
        self.rotation_delta = rotation_delta
        self.variant = variant
        self.initializer = initializer
        self.activation = activation

        # Define down-projection layer
        assert variant in ["dirac", "geodesic"], "Please choose a layer type from: ['dirac', 'geodesic']."
        self.layer_type = ConvGeodesic if variant == "geodesic" else ConvDirac

        self.down_projection = tf.keras.layers.Dense(units=1, activation="sigmoid")

    def build(self, input_shape):
        signal_shape, _ = input_shape

        # Build intermediate convolution layers
        for idx, dim in enumerate(self.intermediate_dims):
            self.intermediate_convs.append(
                ResNetBlock(
                    amt_templates=self.intermediate_dims[idx],
                    template_radius=self.template_radius,
                    rotation_delta=self.rotation_delta,
                    conv_type=self.variant,
                    activation=self.activation,
                    input_dim=signal_shape[-1] if idx == 0 else self.intermediate_dims[idx - 1],
                    initializer=self.initializer
                )
            )

        # Build pre-bottleneck layer
        self.pre_bottleneck_conv = ResNetBlock(
            amt_templates=self.pre_bottleneck_dim,
            template_radius=self.template_radius,
            rotation_delta=self.rotation_delta,
            conv_type=self.variant,
            activation=self.activation,
            input_dim=signal_shape[-1] if len(self.intermediate_dims) == 0 else self.intermediate_dims[-1],
            initializer=self.initializer
        )

    def call(self, inputs, training=False, **kwargs):
        signal, bc = inputs

        # Propagate through intermediate layers
        for idx, _ in enumerate(self.intermediate_dims):
            signal = self.intermediate_convs[idx]([signal, bc])

        # Initial convolution
        return_signal = self.pre_bottleneck_conv([signal, bc], training=training)

        # Down-projection
        signal_weighting = self.down_projection(return_signal)

        return_signal = signal_weighting * return_signal

        return return_signal, signal_weighting
