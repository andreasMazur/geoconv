from geoconv_examples.mpi_faust.tensorflow.faust_data_set import faust_generator
from geoconv_examples.mpi_faust_segmentation.deepview.DeepView import DeepViewSubClass

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

import tensorflow as tf
import numpy as np
import trimesh


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        # self.fc = collection.get_facecolors()
        # if len(self.fc) == 0:
        #     raise ValueError('Collection must have a facecolor')
        # elif len(self.fc) == 1:
        #     self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        # print(self.ind)
        # self.fc[:, -1] = self.alpha_other
        # self.fc[self.ind, -1] = 1
        # self.collection.set_facecolors(self.fc)
        # self.canvas.draw_idle()
        return self.ind

    def disconnect(self):
        self.lasso.disconnect_events()
        # self.fc[:, -1] = 1
        # self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


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


def interactive_seg_correction(coordinates,
                               all_segments,
                               ground_truth,
                               query_idxs,
                               cmap,
                               amount_classes,
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

    # TODO:
    #   Define class names
    #   class_dict = {
    #         0: "right_arm",
    #         1: "left_arm",
    #         2: "torso",
    #         3: "head",
    #         4: "left_foot",
    #         5: "right_foot",
    #         6: "left_leg",
    #         7: "right_leg",
    #         8: "left_hand",
    #         9: "right_hand"
    #   }
    #     # User interactions
    #     prompt_1 = input("Write this to the output file? [y/n]:").lower()
    #     if prompt_1 == "y":
    #         # Ask for class label
    #         prompt_2 = input("What is the correct class? Give a number 0-9:").lower()
    #         real_class = class_dict[int(prompt_2)]
    #             # Write class label given by user to file
    #         with open(file_name, "a") as f:
    #             f.write(str(shape_idx) + "," + str(query_idx) + "," + real_class + "\n")
    #     elif prompt_1 == "n":
    #         print("okay :(")
    #     # End loop if query was found
    #     break
    #   if not query_found:
    #        raise RuntimeError("Query vertex could not be found.")


def correction_pipeline(model_path, dataset_path):
    # Load model and dataset
    model = tf.keras.models.load_model(model_path)
    dataset = faust_generator(dataset_path, set_type=3, only_signal=False, return_coordinates=True)

    for (signal, bc, coordinates), labels in dataset:
        coordinates = np.array(coordinates)
        embeddings = embed(model, [signal, bc])

        # Determine segments by binning the vertices by their class labels
        all_segments = [coordinates[np.where(labels == x)[0]] for x in range(10)]

        # loading the values needed for visualizing the mesh segment
        def dv_show_seg(vertex_idx, point, pred, gt):
            interactive_seg_correction(
                coordinates=coordinates,
                all_segments=all_segments,
                ground_truth=gt,
                query_idxs=vertex_idx,
                cmap=cmap,
                file_name="corrected_labels.csv"
            )

        # --- Deep View Parameters ----
        classes = np.arange(np.unique(labels).shape[0])
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
            data_viz=dv_show_seg,
            metric=metric,
            disc_dist=disc_dist
        )

        imcnn_deepview.add_samples(embeddings, labels)
        imcnn_deepview.show()
