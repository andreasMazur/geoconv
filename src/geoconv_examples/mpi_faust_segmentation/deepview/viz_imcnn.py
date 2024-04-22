import tensorflow as tf
import numpy as np
import trimesh
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

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
    return signal

def get_embeddings(dataset,model, shape):
    embeddings = []
    coords = []
    gts = []
    i = 0
    for (signal, bc, coord), gt in dataset:
        if i == shape:
            embeddings.append(embed(model, [signal, bc]))
            coords.append(coord)
            gts.append(gt)
            return embeddings, gt, coords
        i += 1
    return None

def pred_wrapper(x, model):
    """
    Model classifier over the sample given to deepview. This function will be pass to Deepview.
    :param x: sample
    :return: Probability vector
    """
    logits = model.output_dense(x)
    output = tf.nn.softmax(logits)
    return output

def interactive_seg_correction(shape_idx, coordinates, all_segments, ground_truth, query_idxs,
                               idxs_preds, idxs_labels,cmap, file_name="corrected_labels.csv"):
    """Interactively correct the segmentation label.

    Parameters
    ----------
    shape_idx: int
        The index of the current shape in the FAUST dataset.
    coordinates: np.ndarray
        The coordinates of the current shape.
    ground_truth: np.ndarray
        The current ground truth segmentation labels for the given shape.
    query_idx: int
        The index of the query vertex.
    file_name: str
        The file name in which to write corrections.
    """
    # Define class names
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

    query_coordinates = coordinates[query_idxs]

    # Assign vertex colors
    unique_segment_labels = np.unique(ground_truth)
    unique_segment_colors = [cmap(c / (10 - 1)) for c in range(10)]  # amount classes + 10
    segments = np.concatenate([all_segments[x] for x in unique_segment_labels], axis=0)
    colors = np.full(shape=(segments.shape[0], 4), fill_value=[1, 1, 1., 1.])
    for idx, qc in enumerate(query_coordinates):
        colors[np.where((qc == segments).all(axis=-1))[0][0]] = unique_segment_colors[ground_truth[idx]]


    # Show mesh
    mesh = trimesh.PointCloud(vertices=segments, colors=colors)
    # print(f"Current class name: {class_dict[ground_truth]} / {ground_truth}")
    mesh.show()

            # User interactions
    prompt_1 = input("Write these points to the output file? [y/n]:").lower()
    if prompt_1 == "y":# Ask for class label
        prompt_2 = input("What is the correct class for this set of points? Give a number 0-9:").lower()
        real_class = class_dict[int(prompt_2)]

                # Write class label given by user to file
        with open(file_name, "a") as f:
            for query_idx in query_idxs:
                f.write(str(shape_idx) + "," + str(query_idx) + "," + str(prompt_2) + "," +  real_class + "\n")
    elif prompt_1 == "n":
        print("okay. Continue selecting. ")



    # # Raise error if query vertex is in no segment
    # if not query_found:
    #     raise RuntimeError("Query vertex could not be found.")


def load_model(model_path):
    """
    loads the model from path
    :param model_path: path to model
    :return: tf.model
    """
    model = tf.keras.models.load_model(model_path)
    return model


