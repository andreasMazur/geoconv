from geoconv.tensorflow.backbone.resnet_block import ResNetBlock

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
        self.down_projection = ResNetBlock(
            amt_templates=1,
            template_radius=self.template_radius,
            rotation_delta=self.rotation_delta,
            conv_type=self.variant,
            activation="linear",
            input_dim=self.pre_bottleneck_dim,
            initializer=self.initializer
        )

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
        for idx in tf.range(len(self.intermediate_dims)):
            signal = self.intermediate_convs[idx]([signal, bc])

        # Initial convolution
        return_signal = self.pre_bottleneck_conv([signal, bc], training=training)

        # Down-projection
        signal_weighting = tf.keras.activations.sigmoid(self.down_projection([return_signal, bc]))
        return_signal = signal_weighting * return_signal

        return return_signal
