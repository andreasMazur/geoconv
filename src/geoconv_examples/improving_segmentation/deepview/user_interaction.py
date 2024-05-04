import numpy as np
import trimesh


def interactive_seg_correction(shape_idx,
                               coordinates,
                               all_segments,
                               ground_truth,
                               query_indices,
                               cmap,
                               class_dict,
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

    # User interactions
    prompt_1 = ""
    while prompt_1 not in ["y", "n"]:
        prompt_1 = input("Write these points to the output file? [y/n]:").lower()

    if prompt_1 == "y":  # Ask for class label
        class_correction = ""
        while class_correction not in [f"{k}" for k in class_dict.keys()]:
            class_correction = input(
                f"What is the correct class for this set of points? Give a number 0-{amount_classes - 1}:"
            ).lower()

        # Write class label given by user to file
        with open(file_name, "a") as f:
            for i, query_idx in enumerate(query_indices):
                f.write(f"{shape_idx},{query_idx},{class_correction}\n")
            print(f"Points corrected.")
    elif prompt_1 == "n":
        print("okay. Continue selecting. ")
