from geoconv.tensorflow.backbone import Covariance
from geoconv.tensorflow.layers import (
    BarycentricCoordinates,
    SpatialDropout,
    ConvZero,
    ConvGeodesic,
    ConvDirac,
    AngularMaxPooling
)
from geoconv.tensorflow.layers import NormalizePointCloud
from geoconv.tensorflow.layers import PointCloudShotDescriptor

import tensorflow as tf


class WarmupAndExpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_rate, decay_steps, warmup_steps):
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def __call__(self, step):
        if step >= self.warmup_steps:
            return self.initial_learning_rate * self.decay_rate ** (
                (step - self.warmup_steps) / self.decay_steps
            )
        else:
            return step / self.warmup_steps * self.initial_learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_rate": self.decay_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
        }


class ModelNetClf(tf.keras.Model):
    def __init__(
        self,
        n_radial,
        n_angular,
        isc_layer_conf,
        template_radius,
        neighbors_for_lrf=32,
        projection_neighbors=10,
        modelnet10=False,
        kernel=None,
        rotation_delta=1,
        pooling="avg",
        exp_lambda=1.0,
        shift_angular=True,
        azimuth_bins=8,
        elevation_bins=6,
        radial_bins=2,
        histogram_bins=6,
        sphere_radius=0.0,
        l1_reg_strength=0.0,
        l2_reg_strength=0.0,
        dropout_rate=0.0
    ):
        super().__init__()

        #############
        # INPUT PART
        #############
        # For centering point clouds
        self.normalize_point_cloud = NormalizePointCloud()

        # For initial vertex signals
        self.neighbors_for_lrf = neighbors_for_lrf
        self.azimuth_bins = azimuth_bins
        self.elevation_bins = elevation_bins
        self.radial_bins = radial_bins
        self.histogram_bins = histogram_bins
        self.sphere_radius = sphere_radius
        self.shot_descriptor = PointCloudShotDescriptor(
            neighbors_for_lrf=self.neighbors_for_lrf,
            azimuth_bins=self.azimuth_bins,
            elevation_bins=self.elevation_bins,
            radial_bins=self.radial_bins,
            histogram_bins=self.histogram_bins,
            sphere_radius=self.sphere_radius,
        )

        # Init barycentric coordinates layer
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.projection_neighbors = projection_neighbors
        self.bc_layer = BarycentricCoordinates(
            n_radial=self.n_radial,
            n_angular=self.n_angular,
            neighbors_for_lrf=self.neighbors_for_lrf,
            projection_neighbors=self.projection_neighbors,
        )

        #################
        # EMBEDDING PART
        #################
        # Determine which layer type shall be used
        self.kernel = kernel
        assert self.kernel in [
            "dirac",
            "geodesic",
            "zero",
        ], "Please choose a layer type from: ['dirac', 'geodesic', 'zero']."
        if self.kernel == "zero":
            self.layer_type = ConvZero
        elif self.kernel == "geodesic":
            self.layer_type = ConvGeodesic
        else:
            self.layer_type = ConvDirac

        # Define embedding architecture
        self.isc_layer_conf = isc_layer_conf
        self.rotation_delta = rotation_delta
        self.template_radius = template_radius
        self.exp_lambda = exp_lambda
        self.shift_angular = shift_angular
        self.isc_layers = []
        self.dropout_rate = dropout_rate
        self.dropout = SpatialDropout(rate=self.dropout_rate)
        for idx, _ in enumerate(self.isc_layer_conf):
            self.isc_layers.append(
                tf.keras.models.Sequential(
                    [
                        self.layer_type(
                            amt_templates=self.isc_layer_conf[idx],
                            template_radius=self.template_radius,
                            rotation_delta=self.rotation_delta,
                            activation="relu",
                            exp_lambda=self.exp_lambda,
                            shift_angular=self.shift_angular,
                            l1_reg_strength=l1_reg_strength,
                            l2_reg_strength=l2_reg_strength
                        ),
                        AngularMaxPooling()
                    ],
                    name=f"isc_layer_{idx}",
                )
            )

        ######################
        # CLASSIFICATION PART
        ######################
        self.pooling = pooling
        assert self.pooling in [
            "cov",
            "max",
            "avg",
        ], "Please set your pooling to either 'cov', 'max' or 'avg'."
        if self.pooling == "cov":
            self.pool = Covariance()
        elif self.pooling == "avg":
            self.pool = tf.keras.layers.GlobalAvgPool1D(data_format="channels_last")
        else:
            self.pool = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")

        # Define classification head
        self.modelnet10 = modelnet10
        self.output_dim = 10 if self.modelnet10 else 40
        self.clf = tf.keras.models.Sequential(
            [
                # tf.keras.layers.Dense(units=128, activation="relu"),
                tf.keras.layers.Dense(units=self.output_dim, activation="linear"),
            ]
        )

    def call(self, inputs, **kwargs):
        # Normalize point-cloud
        coordinates = inputs
        coordinates, aabb = self.normalize_point_cloud(coordinates)

        # Compute barycentric coordinates from 3D coordinates
        bc = self.bc_layer(coordinates)

        # Compute SHOT-descriptor as initial local vertex features
        signal = self.shot_descriptor(coordinates)

        # Compute vertex embeddings
        for idx, _ in enumerate(self.isc_layers):
            signal = self.isc_layers[idx]([signal, bc])
            signal = self.dropout(signal)

        # Pool local surface descriptors into global point-cloud descriptor
        signal = self.pool(signal)

        # Add bounding box information to shape descriptor
        signal = tf.concat([signal, aabb], axis=-1)

        # Return classification of point-cloud descriptor
        return self.clf(signal)

    def get_config(self):
        """Get the configuration dictionary.

        Returns
        -------
        dict:
            The configuration dictionary.
        """
        config = super(ModelNetClf, self).get_config()
        config.update(
            {
                "n_radial": self.n_radial,
                "n_angular": self.n_angular,
                "isc_layer_conf": self.isc_layer_conf,
                "template_radius": self.template_radius,
                "neighbors_for_lrf": self.neighbors_for_lrf,
                "projection_neighbors": self.projection_neighbors,
                "modelnet10": self.modelnet10,
                "kernel": self.kernel,
                "rotation_delta": self.rotation_delta,
                "pooling": self.pooling,
                "azimuth_bins": self.azimuth_bins,
                "elevation_bins": self.elevation_bins,
                "radial_bins": self.radial_bins,
                "histogram_bins": self.histogram_bins,
                "sphere_radius": self.sphere_radius,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        """Re-instantiates the model from the config dictionary.

        Parameters
        ----------
        **kwargs
        config: dict
            The configuration dictionary.

        Returns
        -------
        ConvIntrinsic:
            The layer.
        """
        model = cls(**config)
        model.bc_layer.adapt(template_radius=config["template_radius"])
        return model
