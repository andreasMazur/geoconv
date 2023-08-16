from geoconv.layers.angular_max_pooling import AngularMaxPooling
from geoconv.layers.conv_geodesic import ConvGeodesic
from geoconv.models.intrinsic_model import ImCNN

from tensorflow import keras

import keras_tuner
import gc


class OriginalHyperModel(keras_tuner.HyperModel):

    def __init__(self, signal_dim, kernel_size, amt_splits, amt_gradient_splits, kernel_radius, rotation_delta):
        super().__init__()
        self.signal_dim = signal_dim
        self.kernel_size = kernel_size
        self.kernel_radius = kernel_radius
        self.amt_splits = amt_splits
        self.amt_gradient_splits = amt_gradient_splits
        self.rotation_delta = rotation_delta

    def build(self, hp):
        keras.backend.clear_session()
        gc.collect()

        signal_input = keras.layers.Input(shape=self.signal_dim, name="signal")
        bc_input = keras.layers.Input(shape=(self.kernel_size[0], self.kernel_size[1], 3, 2), name="bc")
        amp = AngularMaxPooling()

        name_0 = "LIN16ReLU"
        signal = keras.layers.Dense(16, activation="relu", name=name_0)(signal_input)

        name_1 = "GC32AMPReLU"
        signal = ConvGeodesic(
            output_dim=32,
            amt_kernel=hp.Int("gc_0_amt_kernel", 1, 3),
            rotation_delta=self.rotation_delta,
            kernel_radius=self.kernel_radius,
            activation="relu",
            splits=self.amt_splits,
            name=name_1
        )([signal, bc_input])
        signal = amp(signal)

        name_2 = "GC64AMPReLU"
        signal = ConvGeodesic(
            output_dim=64,
            amt_kernel=hp.Int("gc_1_amt_kernel", 1, 3),
            rotation_delta=self.rotation_delta,
            kernel_radius=self.kernel_radius,
            activation="relu",
            splits=self.amt_splits,
            name=name_2
        )([signal, bc_input])
        signal = amp(signal)

        name_3 = "GC128AMPReLU"
        signal = ConvGeodesic(
            output_dim=128,
            amt_kernel=hp.Int("gc_2_amt_kernel", 1, 3),
            rotation_delta=self.rotation_delta,
            kernel_radius=self.kernel_radius,
            activation="relu",
            splits=self.amt_splits,
            name=name_3
        )([signal, bc_input])
        signal = amp(signal)

        name_4 = "LIN256ReLU"
        signal = keras.layers.Dense(256, activation="linear", name=name_4)(signal)

        name_5 = "LIN6890ReLU"
        output = keras.layers.Dense(6890, activation="linear", name=name_5)(signal)

        model = ImCNN(
            splits=self.amt_gradient_splits,
            inputs=[signal_input, bc_input],
            outputs=[output]
        )
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = keras.optimizers.Adam(learning_rate=hp.Float("lr", min_value=1e-6, max_value=1e-3))
        model.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])

        return model
