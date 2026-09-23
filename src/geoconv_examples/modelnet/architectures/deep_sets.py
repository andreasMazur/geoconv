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
        """Initializes the Deep Set aggregation network.

        Parameters
        ----------
        local_network_dims: list
            The local network dims. Applied to individual vertex features.
        global_network_dims: list
            The global network dims. Applied to the sum over all vertex features.
        local_activation: Any
            The local activation.
        global_activation: Any
            The global activation.
        *args: tuple
            The args.
        **kwargs: dict
            The kwargs.
        """
        super().__init__(*args, **kwargs)
        self.local_network_dims = local_network_dims
        self.local_activation = local_activation
        self.global_network_dims = global_network_dims
        self.global_activation = global_activation

        # Set in build
        self.local_network = None
        self.global_network = None

    def build(self, input_shape):
        """Builds the local and global feed-forward sub-networks.

        Parameters
        ----------
        input_shape: list
            The input shape.
        """
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
        """Applies Deep Set aggregation to the input signal and mask.

        Parameters
        ----------
        inputs: tf.Tensor
            The inputs tensor.

        Returns
        -------
        tf.Tensor
            The computed tensor.
        """
        signal, mask = inputs
        signal = self.local_network(signal)
        mask = tf.cast(mask[..., None], tf.float32)
        signal = signal * mask
        signal = tf.reduce_sum(signal, axis=-2) / tf.reduce_sum(mask, axis=-2)
        return self.global_network(signal)
