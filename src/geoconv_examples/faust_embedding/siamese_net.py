from geoconv_examples.faust.classifer import FaustVertexClassifier, AMOUNT_VERTICES, SIG_DIM

import tensorflow as tf


def load_siamese(path,
                 template_radius,
                 n_radial,
                 n_angular,
                 isc_layer_dims,
                 variant=None,
                 normalize_input=True,
                 rotation_delta=1):
    model = SiameseIMCNN(
        template_radius,
        isc_layer_dims,
        variant=variant,
        normalize_input=normalize_input,
        rotation_delta=rotation_delta
    )
    model.build(
        input_shape=[
            tf.TensorShape([None, AMOUNT_VERTICES, SIG_DIM]),
            tf.TensorShape([None, AMOUNT_VERTICES, n_radial, n_angular, 3, 2]),
            tf.TensorShape([None, AMOUNT_VERTICES, SIG_DIM]),
            tf.TensorShape([None, AMOUNT_VERTICES, n_radial, n_angular, 3, 2])
        ]
    )
    model.load_weights(path)

    clf = FaustVertexClassifier(
        template_radius,
        isc_layer_dims=isc_layer_dims,
        variant=variant,
        normalize_input=normalize_input,
        rotation_delta=rotation_delta,
        include_clf=True
    )
    clf.build(
        input_shape=[
            tf.TensorShape([None, AMOUNT_VERTICES, SIG_DIM]),
            tf.TensorShape([None, AMOUNT_VERTICES, n_radial, n_angular, 3, 2])
        ]
    )
    for clf_layer, siamese_layer in zip(clf.layers, model.backbone.layers):
        try:
            clf_layer.set_weights(siamese_layer.get_weights())
        except ValueError:
            print(f"Skipping copying of weights for layers: \n{clf_layer} \n{siamese_layer}")
    return clf


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
