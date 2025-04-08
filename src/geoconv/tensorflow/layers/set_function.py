import tensorflow as tf


class SetFunction(tf.keras.layers.Layer):
    """A layer that implements the permutation-invariant function from the Deep Sets paper

    Compare:
    > [Deep Sets](https://arxiv.org/abs/1703.06114)
    > Manzil Zaheer et al.
    """

    def __init__(
        self, phi_units, rho_units, phi_activation="relu", rho_activation="linear"
    ):
        super().__init__()
        self.phi = tf.keras.layers.Dense(phi_units, activation=phi_activation)
        self.rho = tf.keras.layers.Dense(rho_units, activation=rho_activation)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return self.rho(tf.reduce_sum(self.phi(inputs), axis=-2))
