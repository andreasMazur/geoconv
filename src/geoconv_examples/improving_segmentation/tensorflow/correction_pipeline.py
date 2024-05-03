from geoconv_examples.mpi_faust.tensorflow.faust_data_set import faust_generator
from geoconv_examples.improving_segmentation.deepview.deepview import DeepViewSubClass
from geoconv_examples.improving_segmentation.data.model_util import embed, pred_wrapper
from geoconv_examples.improving_segmentation.deepview.interactive_correction import interactive_seg_correction

import numpy as np


def correction_pipeline(model_path, dataset_path, amount_classes=10):
    """"""
    # --- Deep View Parameters ----
    classes = np.arange(amount_classes)
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
    title = 'DeepView and GeoConv'
    # -----------------------------

    # Load model and dataset
    model = tf.keras.models.load_model(model_path)
    dataset = faust_generator(dataset_path, set_type=3, only_signal=False, return_coordinates=True)

    for idx, ((signal, bc, coordinates), labels) in enumerate(dataset):
    coordinates = np.array(coordinates)
    embeddings = embed(model, [signal, bc])

    # Determine segments by binning the vertices by their class labels
    all_segments = [coordinates[np.where(labels == x)[0]] for x in range(10)]

    # loading the values needed for visualizing the mesh segment
    def dv_show_seg(vertex_idx, pred, gt, cmap):
        interactive_seg_correction(
            shape_idx=idx,
            coordinates=coordinates,
            all_segments=all_segments,
            ground_truth=gt,
            query_idxs=vertex_idx,
            cmap=cmap,
            idxs_preds=pred,
            idxs_labels=gt,
            amount_classes=amount_classes,
            file_name="corrected_labels.csv"
        )

    imcnn_deepview = DeepViewSubClass(
        lambda x: pred_wrapper(x, model),
        classes,
        max_samples,
        batch_size,
        data_shape,
        N,
        lam,
        resolution,
        cmap,
        interactive,
        title,
        metric=metric,
        disc_dist=disc_dist,
        data_viz=dv_show_seg
    )
    imcnn_deepview.add_samples(embeddings, labels)
    imcnn_deepview.show()
    imcnn_deepview.close()
