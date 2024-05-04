from geoconv.tensorflow.layers.angular_max_pooling import AngularMaxPooling
from geoconv.tensorflow.layers.conv_geodesic import ConvGeodesic
from geoconv.tensorflow.layers.conv_zero import ConvZero
from geoconv.tensorflow.layers.conv_dirac import ConvDirac

import tensorflow as tf
import keras


class Imcnn(keras.Model):
    def __init__(self,
                 signal_dim,
                 kernel_size,
                 template_radius,
                 layer_conf=None,
                 variant="dirac",
                 segmentation_classes=-1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_dim = signal_dim
        self.kernel_size = kernel_size
        self.template_radius = template_radius

        if variant == "dirac":
            self.layer_type = ConvDirac
        elif variant == "geodesic":
            self.layer_type = ConvGeodesic
        elif variant == "zero":
            self.layer_type = ConvZero
        else:
            raise RuntimeError("Select a layer type from: ['dirac', 'geodesic', 'zero']")

        if layer_conf is None:
            self.output_dims = [96, 256, 384, 384]
            self.rotation_deltas = [1 for _ in range(len(self.output_dims))]
        else:
            self.output_dims, self.rotation_deltas = list(zip(*layer_conf))

        #################
        # Handling Input
        #################
        self.normalize = keras.layers.Normalization(axis=-1, name="input_normalization")
        self.downsize_dense = keras.layers.Dense(64, activation="relu", name="downsize")
        self.downsize_bn = keras.layers.BatchNormalization(axis=-1, name="BN_downsize")

        #############
        # ISC Layers
        #############
        self.isc_layers = []
        self.bn_layers = []
        self.do_layers = []
        self.amp_layers = []
        for idx in range(len(self.output_dims)):
            self.do_layers.append(keras.layers.Dropout(rate=0.2, name=f"DO_layer_{idx}"))
            self.isc_layers.append(
                self.layer_type(
                    amt_templates=self.output_dims[idx],
                    template_radius=self.template_radius,
                    activation="relu",
                    name=f"ISC_layer_{idx}",
                    rotation_delta=self.rotation_deltas[idx]
                )
            )
            self.bn_layers.append(keras.layers.BatchNormalization(axis=-1, name=f"BN_layer_{idx}"))
            self.amp_layers.append(AngularMaxPooling())

        #########
        # Output
        #########
        if segmentation_classes:
            self.output_dense = keras.layers.Dense(segmentation_classes, name="output")
        else:
            self.output_dense = keras.layers.Dense(6890, name="output")

        ###################
        # Training metrics
        ###################
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy = keras.metrics.SparseCategoricalAccuracy()

    def call(self, inputs, **kwargs):
        #################
        # Handling Input
        #################
        signal, bc = inputs
        signal = self.normalize(signal)
        signal = self.downsize_dense(signal)
        signal = self.downsize_bn(signal)

        ###############
        # Forward pass
        ###############
        for idx in range(len(self.output_dims)):
            signal = self.do_layers[idx](signal)
            signal = self.isc_layers[idx]([signal, bc])
            signal = self.amp_layers[idx](signal)
            signal = self.bn_layers[idx](signal)

        #########
        # Output
        #########
        return self.output_dense(signal)

    def test_step(self, data):
        (signal, bc), gt = data
        pred = self([signal, bc], training=False)
        loss = self.compute_loss(y=gt, y_pred=pred)

        # Statistics
        self.loss_tracker.update_state(loss)
        for metric in self.metrics[1:]:  # skip loss
            metric.update_state(gt, pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        (signal, bc), gt = data

        with tf.GradientTape() as tape:
            pred = self([signal, bc], training=True)
            loss = self.compute_loss(y=gt, y_pred=pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Statistics
        self.loss_tracker.update_state(loss)
        for metric in self.metrics[1:]:  # skip loss
            metric.update_state(gt, pred)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # 'reset_states()' called automatically at the start of each training epoch or evaluation
        return [self.loss_tracker, self.accuracy]
