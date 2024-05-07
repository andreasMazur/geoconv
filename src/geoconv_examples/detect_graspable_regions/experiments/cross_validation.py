from geoconv_examples.detect_graspable_regions.partnet_grasp.dataset import PartNetGraspDataset
from geoconv_examples.detect_graspable_regions.training.imcnn import SegImcnn
from geoconv_examples.detect_graspable_regions.training.train_imcnn import train_single_imcnn

import numpy as np
import scipy as sp
import pandas as pd
import torch
import os


DATASET_LENGTH = 100


def partnet_grasp_cross_validation(k, epochs, zip_file, logging_dir, label_changes_path, trained_models=None):
    """Perform cross-validation on PartNet-Grasp and store change, entropy and mis-predictions in a CSV file."""
    # Create logging dir
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Determine splits
    idx_folds = np.split(np.arange(DATASET_LENGTH), indices_or_sections=k)
    splits = []
    for fold in range(k):
        splits.append(
            {
                "train_indices": list(np.array([idx_folds[x] for x in range(k) if x != fold]).flatten()),
                "test_indices": idx_folds[fold]
            }
        )

    # Train training on splits
    test_accuracy, test_loss = [], []
    for split_idx, split in enumerate(splits):
        if trained_models is None:
            adapt_data = PartNetGraspDataset(zip_file, set_type=0, only_signal=True, set_indices=split["train_indices"])
            train_data = PartNetGraspDataset(zip_file, set_type=0, set_indices=split["train_indices"])
            test_data = PartNetGraspDataset(zip_file, set_type=2, set_indices=split["test_indices"])
            model, hist = train_single_imcnn(
                None,
                n_epochs=epochs,
                logging_dir=f"{logging_dir}/imcnn_split_{split_idx}",
                adapt_data=adapt_data,
                train_data=train_data,
                test_data=test_data,
                skip_validation=True
            )
            test_accuracy.append(hist["test_accuracy"][-1])
            test_loss.append(hist["test_loss"][-1])
        else:
            model = SegImcnn(
                adapt_data=PartNetGraspDataset(
                    zip_file, set_type=0, only_signal=True, set_indices=split["train_indices"]
                )
            )
            model.load_state_dict(torch.load(trained_models[split_idx]))

        # Compute entropy of vertex predictions for test partnet_grasp
        test_data = PartNetGraspDataset(zip_file, set_type=2, set_indices=split["test_indices"])
        for mesh_idx, ((signal, bc), gt) in enumerate(test_data):
            # Capture entropies
            pred = sp.special.softmax(model([signal, bc]).detach().numpy(), axis=-1)
            entropy = sp.stats.entropy(pred, axis=-1)

            # Capture correct/incorrect predictions
            pred = (pred.argmax(axis=-1) == gt.detach().numpy()).astype(np.int32)

            # Load effective changes for this mesh
            mesh_idx = split_idx * len(split["test_indices"]) + mesh_idx
            print(f"Current mesh index: {mesh_idx}")
            change_array = np.load(f"{label_changes_path}/mesh_changes_{mesh_idx}.npy")

            # Save correction, entropy and prediction statistics
            np.savetxt(
                f"{logging_dir}/change_entropy_correct_pred_{mesh_idx}.csv",
                np.stack([change_array, entropy, pred], axis=-1),
                delimiter=","
            )


def filter_method(logging_dir):
    """Comparison method."""
    data_indices = np.arange(DATASET_LENGTH)
    selected_misclassifications = np.zeros(len(data_indices))
    recall = np.zeros(len(data_indices))
    precision = np.zeros(len(data_indices))

    for d in range(len(data_indices)):
        # Load data
        data = pd.read_csv(
            f"{logging_dir}/change_entropy_correct_pred_{data_indices[d]}.csv", header=None
        )
        relabeled = data.iloc[:, 0]
        # entropies = data.iloc[:, 1]
        misclassification = data.iloc[:, 2] == 0

        # Check the overlap of mis-classified points with the selected ones
        selected_misclassifications[d] = np.sum((relabeled + misclassification) == 2)  # TPs
        recall[d] = selected_misclassifications[d] / np.sum(relabeled)
        precision[d] = selected_misclassifications[d] / np.sum(misclassification)

    return np.mean(recall), np.std(recall), np.mean(precision), np.std(precision)
