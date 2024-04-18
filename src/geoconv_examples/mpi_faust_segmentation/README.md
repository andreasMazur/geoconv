# How to execute this project

```python
from geoconv_examples.mpi_faust.tensorflow.faust_data_set import faust_generator
from geoconv_examples.mpi_faust_segmentation.deepview.DeepView import DeepViewSubClass

import numpy as np
from geoconv_examples.mpi_faust_segmentation.deepview.viz_imcnn import load_model, get_embeddings, \
    interactive_seg_correction, pred_wrapper


if __name__ == "__main__":
    model = load_model(model_path="path/to/model")
    dataset_path = "path/to/dataset"
    dataset = faust_generator(dataset_path, set_type=3, only_signal=False, return_coordinates=True)

    shape_idx = 0

    #  X is the embeddings, Y is the labels, coord is the coordinates
    X, Y, coord = get_embeddings(dataset, model, shape=shape_idx)
    coord = coord[0]

    X1 = np.array(X[0])[:100]
    Y1 = np.array(Y[:100])

    coordinates = np.array(coord[:100])

    # Determine segments by binning the vertices by their class labels
    all_segments = [coordinates[np.where(Y1 == x)[0]] for x in range(10)]

    # loading the values needed for visualizing the mesh segment
    dv_show_seg = lambda vertex_idx, point, pred, gt: interactive_seg_correction(
        shape_idx=shape_idx,
        coordinates=coordinates,
        all_segments=all_segments,
        ground_truth=gt,
        query_idx=vertex_idx,
        file_name="corrected_labels.csv"
    )

    pred_wrapper_lambda = lambda x: pred_wrapper(
        x=x,
        model=model
    )

    # --- Deep View Parameters ----
    classes = np.arange(len(np.unique(Y)))
    batch_size = 64
    max_samples = 7000
    data_shape = (96,)
    resolution = 100
    N = 10
    lam = 1
    cmap = 'tab10'
    metric = 'euclidean'
    disc_dist = False

    # to make sure deepview.show is blocking,
    # disable interactive mode
    interactive = False
    title = 'movie-reviews BERT'

    imcnn_deepview = DeepViewSubClass(pred_wrapper_lambda, classes, max_samples, batch_size, data_shape,
                              N, lam, resolution, cmap, interactive, title, data_viz=dv_show_seg,
                              metric=metric, disc_dist=disc_dist)

    imcnn_deepview.add_samples(X1, Y1)

    imcnn_deepview.show()
```