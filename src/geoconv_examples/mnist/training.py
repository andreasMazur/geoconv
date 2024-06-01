from geoconv.utils.common import read_template_configurations
from geoconv_examples.mnist.dataset import load_preprocessed_mnist
from geoconv.tensorflow.layers.conv_dirac import ConvDirac
from geoconv.tensorflow.layers.angular_max_pooling import AngularMaxPooling

import keras
import tensorflow as tf
import tensorflow_datasets as tfds


class MNISTClassifier(keras.Model):
    def __init__(self, template_radius):
        super().__init__()
        self.conv = ConvDirac(
            amt_templates=32,
            template_radius=template_radius,
            activation="relu",
            rotation_delta=1
        )
        self.amp = AngularMaxPooling()
        self.output_layer = keras.layers.Dense(10)

    def call(self, inputs, **kwargs):
        signal, bc = [x[0] for x in inputs]
        signal = self.conv([signal, bc])
        signal = self.amp(signal)
        signal = tf.reshape(signal, (1, -1))
        return self.output_layer(signal)


def training(bc_path, k=5):
    # Prepare k-fold cross-validation
    splits = tfds.even_splits("all", n=k)

    # Prepare template configurations
    template_configurations = read_template_configurations(bc_path)

    # Run experiments
    for (n_radial, n_angular, template_radius) in template_configurations:
        for exp_no in range(len(splits)):
            # Load data
            train_data = load_preprocessed_mnist(
                bc_path, n_radial, n_angular, template_radius, set_type=splits[:exp_no] + splits[exp_no+1:]
            )
            val_data = load_preprocessed_mnist(bc_path, n_radial, n_angular, template_radius, set_type=splits[exp_no])

            # Define and compile model
            imcnn = MNISTClassifier(template_radius)
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            imcnn.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

            # Train model
            imcnn.fit(x=train_data, validation_data=val_data, epochs=100)
