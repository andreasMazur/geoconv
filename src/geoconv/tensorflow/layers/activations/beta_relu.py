import tensorflow as tf


class BetaRelu(tf.keras.layers.Layer):
    """Beta-Relu magnitude activation

    Implements beta-ReLU magnitude activation as described in:
    > 3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data
    > Maurice Weiler, Mario Geiger, Max Welling, Wouter Boomsma and Taco Cohen.
    """
    def __init__(self, min_norm=1e-6, *args, **kwargs):
        """Initializes the object.

        Parameters
        ----------
        min_norm: float
            The min norm.
        *args: tuple
            The args.
        **kwargs: dict
            The kwargs.
        """
        super().__init__(*args, **kwargs)
        self.min_norm = min_norm
        self.beta = None

    def build(self, input_shape):
        """Builds the layer weights.

        Parameters
        ----------
        input_shape: list
            The input shape.
        """
        self.beta = self.add_weight(
            name="beta",
            shape=(input_shape[-1] // 2,),
            trainable=True
        )

    @tf.function
    def call(self, inputs):
        """Applies the layer to the inputs.

        Parameters
        ----------
        inputs: tf.Tensor
            The inputs tensor.

        Returns
        -------
        tf.Tensor
            A tensor containing the beta-relu altered inputs.
        """
        input_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, (input_shape[0], input_shape[1], input_shape[2] // 2, 2))
        inputs_norm = tf.maximum(tf.linalg.norm(inputs, axis=-1), self.min_norm)
        inputs = tf.nn.relu(inputs_norm - self.beta)[..., None] * tf.math.divide_no_nan(inputs, inputs_norm[..., None])
        return tf.reshape(inputs, (input_shape[0], input_shape[1], input_shape[2]))
