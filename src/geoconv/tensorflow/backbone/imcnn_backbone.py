from geoconv.tensorflow.layers import ConvDirac
from geoconv.tensorflow.layers import ConvGeodesic
from geoconv.tensorflow.layers import ConvZero
from geoconv.tensorflow.layers import AngularMaxPooling

import tensorflow as tf


class ImcnnBackbone(tf.keras.Model):
    def __init__(self,
                 isc_layer_dims,
                 n_radial,
                 n_angular,
                 template_radius,
                 variant=None,
                 rescale_input_dim=None,
                 normalize=True,
                 dropout_rate=0.2,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Input configuration
        self.kernel_size = (n_radial, n_angular)
        self.template_radius = template_radius
        self.rescale_input_dim = rescale_input_dim

        # Output configuration
        self.isc_layer_dims = isc_layer_dims

        # ISC-layer-type configuration
        if variant is None or variant == "dirac":
            self.layer_type = ConvDirac
        elif variant == "geodesic":
            self.layer_type = ConvGeodesic
        elif variant == "zero":
            self.layer_type = ConvZero
        else:
            raise RuntimeError("Select a layer type from: ['dirac', 'geodesic', 'zero']")

        # Normalize Input
        self.do_normalize = normalize
        if self.do_normalize:
            self.normalize = tf.keras.layers.Normalization(axis=-1, name="input_normalization")
        if self.rescale_input_dim is not None:
            self.downsize_fc = tf.keras.layers.Dense(self.rescale_input_dim, activation="relu", name="FC_downsize")
            self.downsize_bn = tf.keras.layers.BatchNormalization(axis=-1, name="BN_downsize")

        # ISC blocks
        self.isc_layers = []
        self.bn_layers = []
        self.do_layers = []
        self.amp_layers = []
        for idx in range(len(self.isc_layer_dims)):
            self.do_layers.append(tf.keras.layers.Dropout(rate=dropout_rate, name=f"DO_layer_{idx}"))
            self.isc_layers.append(
                self.layer_type(
                    amt_templates=self.isc_layer_dims[idx],
                    template_radius=self.template_radius,
                    activation="relu",
                    name=f"ISC_layer_{idx}",
                    rotation_delta=1
                )
            )
            self.amp_layers.append(AngularMaxPooling())
            self.bn_layers.append(tf.keras.layers.BatchNormalization(axis=-1, name=f"BN_layer_{idx}"))

    def call(self, inputs, **kwargs):
        #################
        # Handling Input
        #################
        signal, bc = inputs
        if self.do_normalize:
            signal = self.normalize(signal)
        if self.rescale_input_dim is not None:
            signal = self.downsize_fc(signal)
            signal = self.downsize_bn(signal)

        ###############
        # Forward pass
        ###############
        for idx in range(len(self.isc_layer_dims)):
            signal = self.do_layers[idx](signal)
            signal = self.isc_layers[idx]([signal, bc])
            signal = self.amp_layers[idx](signal)
            signal = self.bn_layers[idx](signal)

        #########
        # Output
        #########
        return signal
