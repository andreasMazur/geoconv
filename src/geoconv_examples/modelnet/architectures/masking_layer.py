import tensorflow as tf


class MaskingLayer(tf.keras.layers.Layer):
    @tf.function
    def call(self, inputs):
        """Applies the vertex mask and expands the batch dimension.
            
        Parameters
        ----------
        inputs: (tf.Tensor, tf.Tensor)
            The signal tensor and a mask tensor.

        Returns
        -------
        tf.Tensor:
            The masked signal tensor.
        """
        signal, mask = inputs
        return tf.expand_dims(signal[mask], axis=0)
