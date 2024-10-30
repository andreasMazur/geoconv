from geoconv.tensorflow.backbone.resnet_block import ResNetBlock
from geoconv.tensorflow.layers.point_cloud_normals import PointCloudNormals
from geoconv.tensorflow.layers.barycentric_coordinates import BarycentricCoordinates

import tensorflow as tf
import tensorflow_probability as tfp


class ResetMetricsAndLosses(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None, **kwargs):
        self.model.triplet_loss_tracker.reset_state()
        self.model.scc_loss_tracker.reset_state()
        self.model.total_loss.reset_state()
        self.model.acc_metric.reset_state()
        for _, metric in self.model.gradient_metrics.items():
            metric.reset_state()


class ShiftPointCloud(tf.keras.layers.Layer):
    @tf.function(jit_compile=True)
    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs, axis=1, keepdims=True)


class Covariance(tf.keras.layers.Layer):
    @tf.function(jit_compile=True)
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        return tf.reshape(tfp.stats.covariance(inputs, sample_axis=1), (input_shape[0], input_shape[-1] ** 2))


class ModelNetClf(tf.keras.Model):
    def __init__(self,
                 n_radial,
                 n_angular,
                 template_radius,
                 isc_layer_dims,
                 neighbors_for_lrf=16,
                 modelnet10=False,
                 variant=None,
                 rotation_delta=1,
                 dropout_rate=0.3,
                 initializer="glorot_uniform",
                 pooling="cov",
                 triplet_alpha=1.0,
                 noise_stddev=1e-3):
        super().__init__()

        #############
        # INPUT PART
        #############
        # Init barycentric coordinates layer
        self.bc_layer = BarycentricCoordinates(
            n_radial=n_radial, n_angular=n_angular, neighbors_for_lrf=neighbors_for_lrf
        )
        self.bc_layer.adapt(template_radius=template_radius)

        # For centering point clouds
        self.center = ShiftPointCloud()
        self.normals = PointCloudNormals(neighbors_for_lrf=neighbors_for_lrf)

        #################
        # EMBEDDING PART
        #################
        # Determine which layer type shall be used
        assert variant in ["dirac", "geodesic"], "Please choose a layer type from: ['dirac', 'geodesic']."

        # Define vertex embedding architecture
        self.isc_layers = []
        for idx, dim in enumerate(isc_layer_dims):
            self.isc_layers.append(
                ResNetBlock(
                    amt_templates=dim,
                    template_radius=template_radius,
                    rotation_delta=rotation_delta,
                    conv_type=variant,
                    activation="relu",
                    input_dim=-1 if idx == 0 else isc_layer_dims[idx - 1],
                    initializer=initializer
                )
            )

        ######################
        # CLASSIFICATION PART
        ######################
        assert pooling in ["cov", "max"], "Please set your pooling to either 'cov' or 'max'."
        if pooling == "cov":
            self.pool = Covariance()
        else:
            self.pool = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")

        # Define classification layer
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.clf = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(16, activation="elu"),
            tf.keras.layers.Dense(units=10 if modelnet10 else 40),
        ])

        # Losses
        self.mse = tf.keras.losses.MeanSquaredError()
        self.scc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.alpha = triplet_alpha

        # Loss tracker
        self.triplet_loss_tracker = tf.keras.metrics.Mean(name="triplet_loss")
        self.scc_loss_tracker = tf.keras.metrics.Mean(name="scc_loss")
        self.total_loss = tf.keras.metrics.Mean(name="total_loss")

        # Accuracy
        self.acc_metric = tf.keras.metrics.Accuracy(name="accuracy")

        # Gradient statistics
        self.gradient_metrics = {}
        self.gradient_statistics = {}

        self.noise = tf.keras.layers.GaussianNoise(stddev=noise_stddev)

    def call(self, inputs, training=False, **kwargs):
        # Shift point-cloud centroid into 0
        coordinates = self.center(inputs)

        # Compute barycentric coordinates from 3D coordinates
        bc = self.bc_layer(coordinates)

        # Compute normals
        signal = self.normals(coordinates)
        signal = tf.concat([coordinates, signal], axis=-1)
        signal = self.noise(signal, training=training)

        # Compute vertex embeddings
        for idx in range(len(self.isc_layers)):
            signal = self.isc_layers[idx]([signal, bc])

        # Get normalized point-cloud embeddings
        signal = self.pool(signal)
        signal = signal / tf.linalg.norm(signal, axis=-1, keepdims=True)

        # Return classification during inference and additionally the embedding during training
        clf_signal = self.dropout(signal)
        return self.clf(clf_signal), signal

    def train_step(self, data):
        point_clouds, labels = data
        anchor, positive, negative = point_clouds[:, :, 0, :], point_clouds[:, :, 1, :], point_clouds[:, :, 2, :]

        with tf.GradientTape() as tape:
            # Get probability distributions:
            # embedding_a, embedding_p, embedding_n: (batch, vertices, feature_dim)
            _, embedding_a = self(anchor, training=True)
            logits_p, embedding_p = self(positive, training=True)
            _, embedding_n = self(negative, training=True)

            # Compute classification loss
            scc_loss = self.scc(labels, logits_p)

            # Compute triplet loss
            anchor_pos_d = embedding_a - embedding_p
            anchor_pos_d = tf.einsum("bf,bf->b", anchor_pos_d, anchor_pos_d)
            anchor_neg_d = embedding_a - embedding_n
            anchor_neg_d = tf.einsum("bf,bf->b", anchor_neg_d, anchor_neg_d)

            # Mask for semi-hard triplets
            mask = tf.cast(
                tf.math.logical_and(anchor_pos_d < anchor_neg_d, anchor_neg_d < anchor_pos_d + self.alpha), tf.float32
            )
            triplet_loss = tf.math.maximum(
                tf.reduce_sum(mask * (anchor_pos_d - anchor_neg_d + self.alpha)), tf.constant(0.)
            )

            total_loss = scc_loss + triplet_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Clip gradients
        gradients = [tf.clip_by_norm(g, 0.2, axes=[-1]) for g in gradients]

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Capture gradient statistics
        gradients = zip(
            [t.handle._name for t in trainable_vars], [tf.reduce_mean(tf.linalg.norm(g, axis=-1)) for g in gradients]
        )
        gradients = {k: float(v.numpy()) for k, v in gradients}
        for name, gradient_norm in gradients.items():
            if name in self.gradient_metrics.keys():
                self.gradient_metrics[name].update_state(gradient_norm)
                self.gradient_statistics[name] = float(self.gradient_metrics[name].result().numpy())
            else:
                # Init
                self.gradient_metrics[name] = tf.keras.metrics.Mean(name=f"{name}_mean")
                self.gradient_metrics[name].update_state(gradient_norm)
                self.gradient_statistics[name] = float(self.gradient_metrics[name].result().numpy())

        # Compute metrics
        self.triplet_loss_tracker.update_state(triplet_loss)
        self.scc_loss_tracker.update_state(scc_loss)
        self.total_loss.update_state(total_loss)

        logits_p = tf.nn.softmax(logits_p, axis=-1)
        self.acc_metric.update_state(labels, tf.expand_dims(tf.math.argmax(logits_p, axis=-1), axis=-1))

        return {
            "triplet_loss": self.triplet_loss_tracker.result(),
            "scc_loss": self.scc_loss_tracker.result(),
            "loss": self.total_loss.result(),
            "accuracy": self.acc_metric.result()
        }

    def test_step(self, data):
        point_clouds, labels = data
        anchor, positive, negative = point_clouds[:, :, 0, :], point_clouds[:, :, 1, :], point_clouds[:, :, 2, :]

        _, embedding_a = self(anchor, training=False)
        logits_p, embedding_p = self(positive, training=False)
        _, embedding_n = self(negative, training=False)

        # Compute classification loss
        scc_loss = self.scc(labels, logits_p)

        # Compute triplet loss
        anchor_pos_d = embedding_a - embedding_p
        anchor_pos_d = tf.einsum("bf,bf->b", anchor_pos_d, anchor_pos_d)
        anchor_neg_d = embedding_a - embedding_n
        anchor_neg_d = tf.einsum("bf,bf->b", anchor_neg_d, anchor_neg_d)

        # Don't mask for semi-hard triplets during testing
        triplet_loss = tf.math.maximum(tf.reduce_sum(anchor_pos_d - anchor_neg_d + self.alpha), tf.constant(0.))

        total_loss = scc_loss + triplet_loss

        # Compute metrics
        self.triplet_loss_tracker.update_state(triplet_loss)
        self.scc_loss_tracker.update_state(scc_loss)
        self.total_loss.update_state(total_loss)

        logits_p = tf.nn.softmax(logits_p, axis=-1)
        self.acc_metric.update_state(labels, tf.expand_dims(tf.math.argmax(logits_p, axis=-1), axis=-1))

        return {
            "triplet_loss": self.triplet_loss_tracker.result(),
            "scc_loss": self.scc_loss_tracker.result(),
            "loss": self.total_loss.result(),
            "accuracy": self.acc_metric.result()
        }
