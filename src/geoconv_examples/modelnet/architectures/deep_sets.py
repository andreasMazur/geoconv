import tensorflow as tf


class DeepSet(tf.keras.layers.Layer):
    """Implements the permutation invariant operation of Deep Sets.

    For more information, see:
    > [Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)
    > Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, Alexander J Smola
    """
    def __init__(self,
                 local_network_dims,
                 global_network_dims,
                 local_activation="relu",
                 global_activation="relu",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.local_network_dims = local_network_dims
        self.local_activation = local_activation
        self.global_network_dims = global_network_dims
        self.global_activation = global_activation

        # Set in build
        self.local_network = None
        self.global_network = None

    def build(self, input_shape):
        super().build(input_shape)
        self.local_network = tf.keras.Sequential(
            [tf.keras.layers.Dense(x, activation=self.local_activation) for x in self.local_network_dims],
            name="local_network"
        )
        self.global_network = tf.keras.Sequential(
            [tf.keras.layers.Dense(x, activation=self.global_activation) for x in self.global_network_dims],
            name="global_network"
        )

    @tf.function
    def call(self, inputs):
        signal, mask = inputs
        signal = self.local_network(signal)
        mask = tf.cast(mask[..., None], tf.float32)
        signal = signal * mask
        signal = tf.reduce_sum(signal, axis=-2) / tf.reduce_sum(mask, axis=-2)
        return self.global_network(signal)
