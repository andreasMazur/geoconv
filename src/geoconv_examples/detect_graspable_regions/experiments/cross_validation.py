from geoconv_examples.detect_graspable_regions.partnet_grasp.dataset import PartNetGraspDataset
from geoconv_examples.detect_graspable_regions.training.imcnn import SegImcnn
from geoconv_examples.detect_graspable_regions.training.train_imcnn import train_single_imcnn

import torch
import numpy as np
import scipy as sp
import os


def partnet_cross_validation(k, epochs, zip_file, logging_dir, label_changes_path, trained_models=None):
    """Perform cross-validation on PartNet-Grasp and store change, entropy and mis-predictions in a CSV file."""
    # Create logging dir
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Determine splits
    dataset_length = 100
    idx_folds = np.split(np.arange(dataset_length), indices_or_sections=k)
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
