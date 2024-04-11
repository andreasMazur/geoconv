import numpy as np
import trimesh


def interactive_seg_correction(shape_idx, coordinates, ground_truth, query_idx, file_name="corrected_labels.csv"):
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
    coordinates, ground_truth = np.array(coordinates), np.array(ground_truth)

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

    # Load the coordinates of the query vertex
    query_coordinates = coordinates[query_idx]

    # Determine segments by binning the vertices by their class labels
    all_segments = [coordinates[np.where(ground_truth == x)[0]] for x in range(10)]

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
            print(f"Current class name: {class_dict[ground_truth[query_idx]]} / {ground_truth[query_idx]}")
            mesh.show()

            # User interactions
            prompt_1 = input("Write this to the output file? [y/n]:").lower()
            if prompt_1 == "y":
                # Ask for class label
                prompt_2 = input("What is the correct class? Give a number 0-9:").lower()
                real_class = class_dict[int(prompt_2)]

                # Write class label given by user to file
                with open(file_name, "a") as f:
                    f.write(str(shape_idx) + "," + str(query_idx) + "," + real_class)
            elif prompt_1 == "n":
                print("okay :(")

            # End loop if query was found
            break

    # Raise error if query vertex is in no segment
    if not query_found:
        raise RuntimeError("Query vertex could not be found.")
