import numpy as np
import trimesh


def interactive_seg_correction(shape_idx,
                               coordinates,
                               all_segments,
                               ground_truth,
                               query_indices,
                               cmap,
                               pred_indices,
                               label_indices,
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
    for idx, qc in enumerate(coordinates[query_indices]):
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
            for i, query_idx in enumerate(query_indices):
                f.write(
                    f"{shape_idx}, "
                    f"{query_idx}, "
                    f"{prompt_2}, "
                    f"{real_class}, "
                    f"{class_dict[pred_indices[i]]}, "
                    f"{class_dict[label_indices[i]]}\n"
                )
            print(f"Points corrected.")
    elif prompt_1 == "n":
        print("okay. Continue selecting. ")
