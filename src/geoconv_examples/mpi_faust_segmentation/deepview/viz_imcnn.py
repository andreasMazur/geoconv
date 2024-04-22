from geoconv_examples.mpi_faust.tensorflow.faust_data_set import faust_generator
from geoconv_examples.mpi_faust_segmentation.deepview.DeepView import DeepViewSubClass

import tensorflow as tf
import numpy as np
import trimesh
from matplotlib import pyplot as plt


def embed(model, inputs):
    #################
    # Handling Input
    #################
    signal, bc = inputs
    signal = model.normalize(signal)
    signal = model.downsize_dense(signal)
    signal = model.downsize_bn(signal)

    ###############
    # Forward pass
    ###############
    for idx in range(len(model.do_layers)):
        signal = model.do_layers[idx](signal)
        signal = model.isc_layers[idx]([signal, bc])
        signal = model.amp_layers[idx](signal)
        signal = model.bn_layers[idx](signal)

    #########
    # Output
    #########
    return np.array(signal)


def pred_wrapper(x, model):
    """
    Model classifier over the sample given to deepview. This function will be pass to Deepview.
    :param x: sample
    :return: Probability vector
    """
    logits = model.output_dense(x)
    output = tf.nn.softmax(logits)
    return output


def interactive_seg_correction(shape_idx,
                               coordinates,
                               all_segments,
                               ground_truth,
                               query_idxs,
                               cmap,
                               idxs_preds,
                               idxs_labels,
                               amount_classes=10,
                               file_name="corrected_labels.csv"):
    """Interactively correct the segmentation label."""

    ##########
    # 3D Plot
    ##########
    # Define colors
    segments = np.concatenate([all_segments[x] for x in np.unique(ground_truth)], axis=0)
    colors = np.full(shape=(segments.shape[0], 4), fill_value=[1., 1., 1., 1.])  # default color

    unique_segment_colors = [cmap(c / (amount_classes - 1)) for c in range(amount_classes)]
    for idx, qc in enumerate(coordinates[query_idxs]):
        colors[np.where((qc == segments).all(axis=-1))[0][0]] = unique_segment_colors[ground_truth[idx]]

    # Show mesh
    mesh = trimesh.PointCloud(vertices=segments, colors=colors)
    mesh.show()

    class_dict = {
        0: "right_arm",
        1: "left_arm",
        2: "torso",
        3: "head",
        4: "left_foot",
        5: "right_foot",
        6: "left_leg",
        7: "right_leg",
        8: "left_hand",
        9: "right_hand"
    }

    # User interactions
    prompt_1 = ""
    while prompt_1 not in ["y", "n"]:
        prompt_1 = input("Write these points to the output file? [y/n]:").lower()

    if prompt_1 == "y":  # Ask for class label
        prompt_2 = ""
        while prompt_2 not in [f"{k}" for k in class_dict.keys()]:
            prompt_2 = input("What is the correct class for this set of points? Give a number 0-9:").lower()

        # Write class label given by user to file
        real_class = class_dict[int(prompt_2)]
        with open(file_name, "a") as f:
            for i, query_idx in enumerate(query_idxs):
                f.write(
                    f"{shape_idx}, "
                    f"{query_idx}, "
                    f"{prompt_2}, "
                    f"{real_class}, "
                    f"{class_dict[idxs_preds[i]]}, "
                    f"{class_dict[idxs_labels[i]]}\n"
                )
            print(f"Points corrected.")
    elif prompt_1 == "n":
        print("okay. Continue selecting. ")


def correction_pipeline(model_path, dataset_path, amount_classes=10):
    # Load model and dataset
    model = tf.keras.models.load_model(model_path)
    dataset = faust_generator(dataset_path, set_type=3, only_signal=False, return_coordinates=True)

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
    title = 'movie-reviews BERT'



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
