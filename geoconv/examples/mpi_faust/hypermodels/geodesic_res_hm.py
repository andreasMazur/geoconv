from geoconv.layers.angular_max_pooling import AngularMaxPooling
from geoconv.layers.conv_geodesic import ConvGeodesic
from geoconv.models.intrinsic_model import ImCNN

from tensorflow import keras

import keras_tuner
import gc


class GeoResHyperModel(keras_tuner.HyperModel):

    def __init__(self,
                 signal_dim,
                 kernel_size,
                 amt_target_nodes,
                 amt_convolutions,
                 amt_splits,
                 amt_gradient_splits,
                 kernel_radius,
                 rotation_delta):
        super().__init__()
        self.signal_dim = signal_dim
        self.kernel_size = kernel_size
        self.kernel_radius = kernel_radius
        self.amt_target_nodes = amt_target_nodes
        self.amt_convolutions = amt_convolutions
        self.amt_splits = amt_splits
        self.amt_gradient_splits = amt_gradient_splits
        self.rotation_delta = rotation_delta

    def build(self, hp):
        keras.backend.clear_session()
        gc.collect()

        signal_input = keras.layers.Input(shape=self.signal_dim, name="signal")
        bc_input = keras.layers.Input(shape=(self.kernel_size[0], self.kernel_size[1], 3, 2), name="bc")
        amp = AngularMaxPooling()

        signal_in = ConvGeodesic(
            output_dim=128,
            amt_kernel=1,
            rotation_delta=self.rotation_delta,
            kernel_radius=self.kernel_radius,
            activation="relu",
            splits=self.amt_splits,
            name="gc_0"
        )([signal_input, bc_input])
        signal_in = amp(signal_in)
        for idx in range(1, self.amt_convolutions):
            signal_1 = ConvGeodesic(
                output_dim=128,
                amt_kernel=1,
                rotation_delta=self.rotation_delta,
                kernel_radius=self.kernel_radius,
                activation="relu",
                splits=self.amt_splits,
                name=f"gc_{idx}_1"
            )([signal_in, bc_input])
            signal_1 = amp(signal_1)

            signal_2 = ConvGeodesic(
                output_dim=128,
                amt_kernel=1,
                rotation_delta=self.rotation_delta,
                kernel_radius=self.kernel_radius,
                activation="relu",
                splits=self.amt_splits,
                name=f"gc_{idx}_2"
            )([signal_1, bc_input])
            signal_2 = amp(signal_2)

            signal_in = keras.layers.Add()([signal_1, signal_2])
            signal_in = keras.layers.ReLU()(signal_in)

        output = keras.layers.Dense(self.amt_target_nodes)(signal_in)

        model = ImCNN(
            splits=self.amt_gradient_splits,
            inputs=[signal_input, bc_input],
            outputs=[output]
        )
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = keras.optimizers.Adam(learning_rate=hp.Float("lr", min_value=1e-5, max_value=1e-1))
        model.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])

        return model
