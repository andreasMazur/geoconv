# Example usage of GeoConv

In this example we present the usage of **GeoConv** by preprocessing- and training on the
[FAUST dataset](http://faust.is.tue.mpg.de/). We will train a geodesic convolutional neural network
to detect point-correspondences between a query- and a reference-mesh. If you want to execute this script, you have to
download the data from their [official website](http://faust.is.tue.mpg.de/).

## Preprocessing

Once the original data is downloaded, we can start to convert it into a format the layer can
work with. We are particularly interested in the PLY-files located in the registrations-folder:
```bash
/home/user/MPI-FAUST/training/registrations
```
Let's take a look on how a possible preprocessing-script could look like:

```python
from layers.examples.mpi_faust.preprocess import preprocess

if __name__ == "__main__":
    path_to_registrations = "/home/user/MPI-FAUST/training/registrations"
    preprocess(
        directory=path_to_registrations,
        target_dir=f"/home/user/preprocessed_registrations",
        reference_mesh=f"{path_to_registrations}/tr_reg_000.ply"
    )
```
The preprocessing function:
1. Loads the original data
2. Computes ground-truth values for supervised learning
3. Calculates local GPC-systems and Barycentric coordinates of the sub-meshes
4. Zips the final dataset

Note the meanings of the arguments:
- directory: The path to the FAUST-dataset
- target_dir: The path to the zip-file that will contain the preprocessed data
- reference_mesh: The path to the mesh which will be used to compute ground truth values

## Neural Network Definition

We model a neural network by simply including the geodesic convolution provided by **GeoConv** into a Tensorflow
model. For example:

```python
from tensorflow.keras.layers import InputLayer, Dense, Normalization
from geoconv.layers.geodesic_conv import ConvGeodesic
from geoconv.layers.angular_max_pooling import AngularMaxPooling 


def define_model(amt_nodes, kernel_size, output_dim=6890, lr=.00045):
    signal_input = InputLayer(shape=(amt_nodes, 3))
    bary_input = InputLayer(shape=(amt_nodes, kernel_size[1], kernel_size[0], 3, 2))
    amp = AngularMaxPooling()
    
    signal = Normalization()(signal_input)
    signal = ConvGeodesic(
        output_dim=64, amt_kernel=2, activation="relu", rotation_delta=2
    )([signal, bary_input])
    signal = amp(signal)
    logits = Dense(output_dim)(signal)

    model = tf.keras.Model(inputs=[signal_input, bary_input], outputs=[logits])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])
    model.summary()
    return model
```

## Dataset

We want to define a dataset that allows us to load single meshes into memory instead of the
entire dataset at once. An example-implementation can be looked up in ``tf_dataset.py``

## Finding Hyperparameters

Instead of implementing our own hyperparameter search, we make use of the Hyperband-implementation from the
[Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner). Take a look on `hypertuning.py` for
further details. One can start the hypertuning with a simple script:

```python
from layers.examples.mpi_faust.hypertuning import faust_hypertuning

if __name__ == "__main__":
    faust_hypertuning(
        path_preprocessed_dataset="../preprocessed_smaller_shot_v3.zip",
        run_id=0
    )
```

Once optimal hyperparameters are found, you can copy them and train you final network with them.
