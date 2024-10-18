from geoconv.tensorflow.layers.barycentric_coordinates import BarycentricCoordinates
from geoconv.tensorflow.layers.tangent_projections import TangentProjections
from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.conv_geodesic import ConvGeodesic
from geoconv.tensorflow.layers.pooling.angular_max_pooling import AngularMaxPooling

import tensorflow as tf
import tensorflow_probability as tfp


ADAPTION_MEANS = {
    5: tf.constant([[[
        1.3662160e-05, 4.8941374e-04, -4.3671846e-02, 1.2019062e+01,
        1.1442985e-02, 1.8237881e+01, -8.5033655e-02, 2.2085234e+01,
        -2.9914480e-01, 2.4544855e+01
    ]]]),
    15: tf.constant([[[
        -3.7189561e-06, 2.8619400e-04, 2.0740835e-03, 6.5275993e+00, -5.2857199e-03,
        1.0554636e+01, 5.7773598e-02, 1.3915789e+01, 6.0970016e-02, 1.6840271e+01,
        5.7036400e-02, 1.9415283e+01, 4.4104282e-02, 2.1721455e+01, 1.1490919e-01,
        2.3646830e+01, 9.9816084e-02, 2.5365578e+01, 1.7695330e-02, 2.6861437e+01,
        5.4662503e-02, 2.8197876e+01, -2.4771576e-03, 2.9295244e+01, 3.9666731e-02,
        3.0319246e+01, 8.8442909e-03, 3.1160933e+01, -2.6727753e-02, 3.1874565e+01
    ]]]),
    25: tf.constant([[[
        -5.53380369e-06, 1.90778737e-04,  1.97725073e-02,  4.93415213e+00,
        1.42620562e-03,  8.10868645e+00,  4.97531630e-02,  1.08360844e+01,
        8.53963494e-02,  1.33628311e+01,  6.83577135e-02,  1.56888809e+01,
        7.06744939e-02,  1.78867855e+01,  1.04257196e-01,  1.99030857e+01,
        1.13290377e-01,  2.18836002e+01,  1.15719296e-01,  2.36793556e+01,
        1.47680312e-01,  2.52640400e+01,  1.24614999e-01,  2.68610687e+01,
        1.04808889e-01,  2.83164883e+01,  1.17059402e-01,  2.97065563e+01,
        1.23031959e-01,  3.09217911e+01,  1.59771517e-01,  3.20935898e+01,
        1.42839745e-01,  3.31394424e+01,  1.06773883e-01,  3.41286392e+01,
        1.41648248e-01,  3.50414619e+01,  1.31660700e-01,  3.58717003e+01,
        8.15364346e-02,  3.66046982e+01,  9.47969705e-02,  3.73758240e+01,
        1.70198575e-01,  3.80099182e+01,  9.96517837e-02,  3.85564041e+01,
        3.63248885e-02,  3.91208420e+01
    ]]])
}
ADAPTION_VARS = {
    5: tf.constant([[[
        3.5439102e-06, 4.8255580e-04, 4.5431919e+00, 2.8862869e+05,
        1.6574842e+00, 6.6457681e+05, 1.6537245e+01, 9.7453756e+05,
        1.8162376e+02, 1.2037026e+06
    ]]]),
    15: tf.constant([[[
        3.3091978e-06, 1.6750646e-04, 8.0988622e-01, 8.5135219e+04, 1.5621265e+00,
        2.2258020e+05, 8.8128710e+00, 3.8691278e+05, 1.0180468e+01, 5.6662244e+05,
        9.8600769e+00, 7.5315806e+05, 7.8372045e+00, 9.4270162e+05, 3.0926426e+01,
        1.1172330e+06, 2.5045353e+01, 1.2855439e+06, 6.3646226e+00, 1.4416391e+06,
        1.2326629e+01, 1.5886534e+06, 6.9686499e+00, 1.7147079e+06, 1.0755516e+01,
        1.8366778e+06, 8.4164190e+00, 1.9400888e+06, 1.0344138e+01, 2.0299696e+06
    ]]]),
    25: tf.constant([[[
        2.6387743e-06, 7.5645636e-05, 1.5944911e+00, 4.8643973e+04, 1.5394777e+00,
        1.3137289e+05, 7.1381311e+00, 2.3461023e+05, 1.7390236e+01, 3.5677588e+05,
        1.2764080e+01, 4.9179309e+05, 1.3990337e+01, 6.3923756e+05, 2.6313311e+01,
        7.9148175e+05, 3.0809639e+01, 9.5683850e+05, 3.2459446e+01, 1.1203088e+06,
        4.9861732e+01, 1.2752761e+06, 3.7867184e+01, 1.4416035e+06, 2.9356018e+01,
        1.6020500e+06, 3.5361862e+01, 1.7632012e+06, 3.8785149e+01, 1.9104032e+06,
        6.0088100e+01, 2.0579319e+06, 5.0474060e+01, 2.1942720e+06, 3.3036228e+01,
        2.3272252e+06, 5.0901489e+01, 2.4533588e+06, 4.6065742e+01, 2.5710048e+06,
        2.5323076e+01, 2.6771502e+06, 3.0546257e+01, 2.7911235e+06, 7.1108833e+01,
        2.8866248e+06, 3.3671101e+01, 2.9702370e+06, 1.7075489e+01, 3.0578408e+06
    ]]])
}


class Covariance(tf.keras.layers.Layer):
    @tf.function
    def call(self, inputs):
        def compute_cov(x):
            x = tfp.stats.covariance(x, sample_axis=1)
            x_shape = tf.shape(x)
            return tf.reshape(x, [x_shape[0], x_shape[1] * x_shape[2]])
        return tf.map_fn(compute_cov, inputs)


class ModelNetClf(tf.keras.Model):
    def __init__(self,
                 n_neighbors,
                 n_radial,
                 n_angular,
                 template_radius,
                 isc_layer_dims,
                 modelnet10=False,
                 variant=None,
                 rotation_delta=1,
                 dropout_rate=0.3):
        super().__init__()

        #############
        # INPUT PART
        #############
        # Init barycentric coordinates layer
        self.bc_layer = BarycentricCoordinates(
            n_radial=n_radial,
            n_angular=n_angular,
            n_neighbors=n_neighbors,
            template_scale=None
        )
        self.bc_layer.adapt(template_radius=template_radius)

        # Input normalization (adapted to ModelNet40)
        assert n_neighbors in [5, 15, 25], "Normalization values have only been computed for 5, 15 or 25 neighbors."
        self.normalize = tf.keras.layers.Normalization(
            axis=-1,
            name="input_normalization",
            mean=ADAPTION_MEANS[n_neighbors],
            variance=ADAPTION_VARS[n_neighbors]
        )

        # Tangent projections
        self.projection_layer = TangentProjections(n_neighbors=n_neighbors)

        #################
        # EMBEDDING PART
        #################
        # Determine which layer type shall be used
        assert variant in ["dirac", "geodesic"], "Please choose a layer type from: ['dirac', 'geodesic']."
        self.layer_type = ConvGeodesic if variant == "geodesic" else ConvDirac

        # Define vertex embedding architecture
        self.isc_layers, self.bn_layers = [], []
        for dim in isc_layer_dims:
            self.isc_layers.append(
                self.layer_type(
                    amt_templates=dim,
                    template_radius=template_radius,
                    activation="elu",
                    name="ISC",
                    rotation_delta=rotation_delta,
                    initializer="glorot_uniform"
                )
            )
            self.bn_layers.append(tf.keras.layers.BatchNormalization(axis=-1, name="batch_normalization"))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.amp = AngularMaxPooling()

        ######################
        # CLASSIFICATION PART
        ######################
        self.pool = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")

        # Define classification layer
        self.clf = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, activation="elu"),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Dense(256, activation="elu"),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Dense(units=10 if modelnet10 else 40),
        ])

    def call(self, inputs, **kwargs):
        # Compute barycentric coordinates from 3D coordinates
        bc = self.bc_layer(inputs)

        # Project into tangent planes
        signal = self.projection_layer(inputs)
        proj_shape = tf.shape(signal)
        signal = tf.reshape(signal, (proj_shape[0], proj_shape[1], proj_shape[2] * proj_shape[3]))

        # Normalize signal
        signal = self.normalize(signal)

        # Compute vertex embeddings
        for idx in range(len(self.isc_layers)):
            signal = self.dropout(signal)
            signal = self.isc_layers[idx]([signal, bc])
            signal = self.amp(signal)
            signal = self.bn_layers[idx](signal)

        # Global max-pool
        signal = self.pool(signal)

        # Return classification logits
        return self.clf(signal)
