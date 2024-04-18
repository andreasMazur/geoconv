import tensorflow as tf
import numpy as np
import trimesh


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

def interactive_seg_correction(shape_idx, coordinates, all_segments, ground_truth, query_idx, file_name="corrected_labels.csv"):
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

    query_coordinates = coordinates[query_idx]

    # Set default value for query-found
    query_found = False

    for segment in all_segments:
        # Check whether query coordinates are in segment
        query_found = query_coordinates in segment
        if query_found:
            # Colorize segment and assign unique color to query vertex
            colors = np.full(shape=(segment.shape[0], 4), fill_value=[0, 0, 255, 255])
            colors[np.where((query_coordinates == segment).all(axis=-1))[0][0]] = [0, 255, 0, 255]

            # Show mesh
            mesh = trimesh.PointCloud(vertices=segment, colors=colors)
            print(f"Current class name: {class_dict[ground_truth]} / {ground_truth}")
            mesh.show()

            # User interactions
            prompt_1 = input("Write this to the output file? [y/n]:").lower()
            if prompt_1 == "y":
                # Ask for class label
                prompt_2 = input("What is the correct class? Give a number 0-9:").lower()
                real_class = class_dict[int(prompt_2)]

                # Write class label given by user to file
                with open(file_name, "a") as f:
                    f.write(str(shape_idx) + "," + str(query_idx) + "," + real_class + "\n")
            elif prompt_1 == "n":
                print("okay :(")

            # End loop if query was found
            break

    # Raise error if query vertex is in no segment
    if not query_found:
        raise RuntimeError("Query vertex could not be found.")


def load_model(model_path):
    """
    loads the model from path
    :param model_path: path to model
    :return: tf.model
    """
    model = tf.keras.models.load_model(model_path)
    return model


