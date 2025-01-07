from geoconv.tensorflow.backbone.covariance import Covariance
from geoconv.tensorflow.backbone.resnet_block import ResNetBlock
from geoconv.tensorflow.layers.barycentric_coordinates import BarycentricCoordinates
from geoconv.tensorflow.layers.shot_descriptor import PointCloudShotDescriptor

import tensorflow as tf


class ModelNetClf(tf.keras.Model):
    def __init__(self,
                 n_radial,
                 n_angular,
                 template_radius,
                 isc_layer_conf,
                 neighbors_for_lrf=16,
                 modelnet10=False,
                 variant=None,
                 rotation_delta=1,
                 initializer="glorot_uniform",
                 pooling="cov",
                 azimuth_bins=8,
                 elevation_bins=2,
                 radial_bins=2,
                 histogram_bins=11,
                 sphere_radius=0.):
        super().__init__()

        #############
        # INPUT PART
        #############
        # For centering point clouds
        self.shot_descriptor = PointCloudShotDescriptor(
            neighbors_for_lrf=neighbors_for_lrf,
            azimuth_bins=azimuth_bins,
            elevation_bins=elevation_bins,
            radial_bins=radial_bins,
            histogram_bins=histogram_bins,
            sphere_radius=sphere_radius,
        )

        # Init barycentric coordinates layer
        self.bc_layer = BarycentricCoordinates(
            n_radial=n_radial, n_angular=n_angular, neighbors_for_lrf=neighbors_for_lrf
        )
        self.bc_layer.adapt(template_radius=template_radius)

        #################
        # EMBEDDING PART
        #################
        # Determine which layer type shall be used
        assert variant in ["dirac", "geodesic"], "Please choose a layer type from: ['dirac', 'geodesic']."

        # Define embedding architecture
        self.isc_layers = []
        for idx, _ in enumerate(isc_layer_conf):
            self.isc_layers.append(
                ResNetBlock(
                    amt_templates=isc_layer_conf[idx],
                    template_radius=template_radius,
                    rotation_delta=rotation_delta,
                    conv_type=variant,
                    activation="relu",
                    input_dim=-1 if idx == 0 else isc_layer_conf[idx - 1],
                    initializer=initializer
                )
            )

        ######################
        # CLASSIFICATION PART
        ######################
        assert pooling in ["cov", "max", "avg"], "Please set your pooling to either 'cov', 'max' or 'avg'."
        if pooling == "cov":
            self.pool = Covariance()
        elif pooling == "avg":
            self.pool = tf.keras.layers.GlobalAvgPool1D(data_format="channels_last")
        else:
            self.pool = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")

        # Define classification layer
        self.output_dim = 10 if modelnet10 else 40
        self.clf = tf.keras.layers.Dense(units=self.output_dim, activation="linear")

    def call(self, inputs, training=False, **kwargs):
        # Compute SHOT-descriptor as initial local vertex features
        coordinates = inputs
        signal = self.shot_descriptor(coordinates)

        # Compute barycentric coordinates from 3D coordinates
        bc = self.bc_layer(coordinates)

        # Compute vertex embeddings
        for idx, _ in enumerate(self.isc_layers):
            signal = self.isc_layers[idx]([signal, bc])

        # Pool local surface descriptors into global point-cloud descriptor
        signal = self.pool(signal)

        # Return classification of point-cloud descriptor
        return self.clf(signal)
