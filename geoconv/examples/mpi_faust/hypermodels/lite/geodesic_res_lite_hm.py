from geoconv.layers.conv_geodesic import ConvGeodesic
from geoconv.models.intrinsic_model import ImCNN

from tensorflow import keras

import keras_tuner
import gc


class GeoResLiteHyperModel(keras_tuner.HyperModel):

    def __init__(self,
                 signal_dim,
                 kernel_size,
                 amt_target_nodes,
                 amt_convolutions,
                 amt_splits,
                 amt_gradient_splits,
                 kernel_radius,
                 output_dim=128):
        super().__init__()
        self.signal_dim = signal_dim
        self.kernel_size = kernel_size
        self.kernel_radius = kernel_radius
        self.amt_target_nodes = amt_target_nodes
        self.amt_convolutions = amt_convolutions
        self.amt_splits = amt_splits
        self.amt_gradient_splits = amt_gradient_splits
        self.output_dim = output_dim

    def build(self, hp):
        keras.backend.clear_session()
        gc.collect()

        signal_input = keras.layers.Input(shape=self.signal_dim, name="signal")
        bc_input = keras.layers.Input(shape=(self.kernel_size[0], self.kernel_size[1], 3, 2), name="bc")

        signal_in = ConvGeodesic(
            output_dim=self.output_dim,
            amt_templates=1,
            template_radius=self.kernel_radius,
            activation="relu",
            splits=self.amt_splits,
            name="gc_0",
            variant="lite"
        )([signal_input, bc_input])

        for idx in range(1, self.amt_convolutions):
            signal_1 = ConvGeodesic(
                output_dim=self.output_dim,
                amt_templates=1,
                template_radius=self.kernel_radius,
                activation="relu",
                splits=self.amt_splits,
                name=f"gc_{idx}_1",
                variant="lite"
            )([signal_in, bc_input])

            signal_2 = ConvGeodesic(
                output_dim=128,
                amt_templates=1,
                template_radius=self.kernel_radius,
                activation="relu",
                splits=self.amt_splits,
                name=f"gc_{idx}_2",
                variant="lite"
            )([signal_1, bc_input])

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
