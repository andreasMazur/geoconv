from geoconv_examples.faust.classifer import FaustVertexClassifier

import tensorflow as tf


class SiameseIMCNN(tf.keras.Model):

    def __init__(self,
                 template_radius,
                 isc_layer_dims,
                 variant=None,
                 normalize_input=True,
                 rotation_delta=1):
        super().__init__()
        self.backbone = FaustVertexClassifier(
            template_radius,
            isc_layer_dims=isc_layer_dims,
            variant=variant,
            normalize_input=normalize_input,
            rotation_delta=rotation_delta,
            include_clf=False
        )

    def call(self, inputs, **kwargs):
        signal_ref, bc_ref, signal, bc = inputs

        embedding_ref = tf.nn.softmax(self.backbone([signal_ref, bc_ref]), axis=-1)
        embedding = tf.nn.softmax(self.backbone([signal, bc]), axis=-1)

        return tf.stack([embedding_ref, embedding], axis=0)
