import tensorflow as tf
import keras


class ISCResidual(keras.Model):
    def __init__(self,
                 first_isc,
                 second_isc,
                 pool,
                 template_radius,
                 layer_conf=None,
                 splits=1,
                 activation="relu",
                 fit_dim=False):
        super().__init__()

        if layer_conf is None:
            self.output_dims = [64, 64]
            self.rotation_deltas = [1, 1]
        else:
            self.output_dims, self.rotation_deltas = list(zip(*layer_conf))

        self.pool = pool()
        self.first_isc = first_isc(
            amt_templates=self.output_dims[0],
            template_radius=template_radius,
            activation=activation,
            name="residual_1",
            splits=splits,
            rotation_delta=self.rotation_deltas[0]
        )
        self.second_isc = second_isc(
            amt_templates=self.output_dims[1],
            template_radius=template_radius,
            activation="linear",
            name="residual_2",
            splits=splits,
            rotation_delta=self.rotation_deltas[1]
        )

        self.fit_dim = fit_dim
        if self.fit_dim:
            self.third_isc = first_isc(
                amt_templates=self.output_dims[0],
                template_radius=template_radius,
                activation=activation,
                name="residual_1",
                splits=splits,
                rotation_delta=self.rotation_deltas[0]
            )

        self.add = keras.layers.Add()
        self.activation = keras.layers.Activation(activation)

    def call(self, inputs, *args, **kwargs):
        entry_signal, bc = inputs

        signal = self.first_isc([entry_signal, bc])
        signal = self.pool(signal)

        signal = self.second_isc([signal, bc])
        signal = self.pool(signal)

        if self.fit_dim:
            entry_signal = self.third_isc([entry_signal, bc])
            entry_signal = self.pool(entry_signal)

        return self.activation(self.add([entry_signal, signal]))
