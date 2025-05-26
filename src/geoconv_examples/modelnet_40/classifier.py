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
        kernel,
        pooling,
        neighbors_for_lrf,
        projection_neighbors,
        azimuth_bins,
        elevation_bins,
        radial_bins,
        histogram_bins,
        sphere_radius,
        n_radial,
        n_angular,
        exp_lambda,
        shift_angular,
        template_scale,
        isc_layer_conf,
        rotation_delta,
        dropout_rate,
        l1_reg_strength,
        l2_reg_strength,
        modelnet10
    ):
        super().__init__()

        # Determine which kernel shall be used
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

        # Determine how to pool local surface descriptors into a global shape descriptor
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

        # Set projection hyperparameters
        self.neighbors_for_lrf = neighbors_for_lrf
        self.projection_neighbors = projection_neighbors

        # Set SHOT descriptor hyperparameters
        self.azimuth_bins = azimuth_bins
        self.elevation_bins = elevation_bins
        self.radial_bins = radial_bins
        self.histogram_bins = histogram_bins
        self.sphere_radius = sphere_radius

        # Set template hyperparameters
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.exp_lambda = exp_lambda
        self.shift_angular = shift_angular
        self.template_scale = template_scale
        self.template_radius = None  # set in adapt()

        # Set architecture hyperparameters
        self.isc_layer_conf = isc_layer_conf
        self.rotation_delta = rotation_delta
        self.dropout_rate = dropout_rate
        self.l1_reg_strength = l1_reg_strength
        self.l2_reg_strength = l2_reg_strength

        # Define classification head
        self.modelnet10 = modelnet10
        self.output_dim = 10 if self.modelnet10 else 40

        # Configure the barycentric coordinates layer
        self.bc_layer = BarycentricCoordinates(
            n_radial=self.n_radial,
            n_angular=self.n_angular,
            neighbors_for_lrf=self.neighbors_for_lrf,
            projection_neighbors=self.projection_neighbors,
        )

        # Initialize the point-cloud normalization layer
        self.normalize_point_cloud = NormalizePointCloud()

        # Configure the SHOT-descriptor layer
        self.shot_layer = PointCloudShotDescriptor(
            neighbors_for_lrf=self.neighbors_for_lrf,
            azimuth_bins=self.azimuth_bins,
            elevation_bins=self.elevation_bins,
            radial_bins=self.radial_bins,
            histogram_bins=self.histogram_bins,
            sphere_radius=self.sphere_radius,
        )

        # Configure the dropout layer
        self.dropout_layer = SpatialDropout(rate=self.dropout_rate)

        # Surface convolutions depend on template radius, which is determined by the BC-layer.
        # Thus, we cannot initialize the ISC layers here, but in the adapt() method.
        self.isc_layers = []

        # Initialize the angular max-pooling layer
        self.amp = AngularMaxPooling()

        # Configure the classification head
        self.clf = tf.keras.layers.Dense(units=self.output_dim, activation="linear")

    def call(self, inputs, **kwargs):
        # Normalize point-cloud
        coordinates = inputs
        coordinates, _ = self.normalize_point_cloud(coordinates)

        # Compute barycentric coordinates from 3D coordinates
        bc = self.bc_layer(coordinates)

        # Compute SHOT-descriptor as initial local vertex features
        signal = self.shot_layer(coordinates)

        # Compute vertex embeddings
        for idx, isc_layer in enumerate(self.isc_layers):
            signal = isc_layer([signal, bc])
            signal = self.amp(signal)
            signal = self.dropout_layer(signal)

        # Pool local surface descriptors into global point-cloud descriptor
        signal = self.pool(signal)

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
                "kernel": self.kernel,
                "pooling": self.pooling,
                "neighbors_for_lrf": self.neighbors_for_lrf,
                "projection_neighbors": self.projection_neighbors,
                "azimuth_bins": self.azimuth_bins,
                "elevation_bins": self.elevation_bins,
                "radial_bins": self.radial_bins,
                "histogram_bins": self.histogram_bins,
                "sphere_radius": self.sphere_radius,
                "n_radial": self.n_radial,
                "n_angular": self.n_angular,
                "exp_lambda": self.exp_lambda,
                "shift_angular": self.shift_angular,
                "template_scale": self.template_scale,
                "template_radius": self.template_radius,
                "isc_layer_conf": self.isc_layer_conf,
                "rotation_delta": self.rotation_delta,
                "dropout_rate": self.dropout_rate,
                "l1_reg_strength": self.l1_reg_strength,
                "l2_reg_strength": self.l2_reg_strength,
                "modelnet10": self.modelnet10
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
        model.adapt(template_radius=config["template_radius"])
        return model

    def adapt(self, template_radius=None, adapt_data=None):
        # Adapt the BC-layer
        if adapt_data is not None:
            self.template_radius = self.bc_layer.adapt(
                data=adapt_data,
                template_scale=self.template_scale,
                exp_lambda=self.exp_lambda,
                shift_angular=self.shift_angular
            )
        elif template_radius is not None:
            self.template_radius = self.bc_layer.adapt(template_radius=template_radius)
        else:
            raise AssertionError(
                "Please provide either a template_radius or adapt BC-layer of the model to your training data."
            )

        for idx, _ in enumerate(self.isc_layer_conf):
            self.isc_layers.append(
                self.layer_type(
                    amt_templates=self.isc_layer_conf[idx],
                    template_radius=self.template_radius,  # Determined by the BC-layer
                    rotation_delta=self.rotation_delta if idx == 0 else self.n_angular,
                    activation="relu",
                    exp_lambda=self.exp_lambda,
                    shift_angular=self.shift_angular,
                    l1_reg_strength=self.l1_reg_strength,
                    l2_reg_strength=self.l2_reg_strength,
                    name=f"isc_layer_{idx}"
                )
            )
