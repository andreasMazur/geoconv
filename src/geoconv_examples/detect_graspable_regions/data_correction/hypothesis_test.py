from geoconv_examples.detect_graspable_regions.data.dataset import PartNetDataset
from geoconv_examples.detect_graspable_regions.data_correction.convert_partnet import convert_partnet
from geoconv_examples.detect_graspable_regions.train_models.train_imcnn import train_single_imcnn

from pathlib import Path

import os
import scipy as sp


def run_experiment(old_dataset_path,
                   new_dataset_path,
                   csv_path,
                   logging_dir,
                   trials=30,
                   epochs=10):
    """Creates the datasets and starts the training runs.

    Parameters
    ----------
    old_dataset_path: str
        The path to the originally pre-processed FAUST dataset
    csv_path: str
        The path to the CSV-file that contains the DeepView-corrections
    new_dataset_path: str
        The path to the DeepView-corrected dataset
    logging_dir: str
        The path to the logging directory
    trials: int
        The amount of trials (i.e. models to train)
    epochs: int
        The amount of epochs per trial
    """
    # Create logging dir
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Integrate DeepView-corrected labels into the dataset
    new_dataset_path = f"{new_dataset_path}.zip"
    if not Path(new_dataset_path).is_file():
        convert_partnet(old_data_path=old_dataset_path, new_data_path=new_dataset_path, csv_path=csv_path)
    else:
        print(f"Found converted dataset file: '{new_dataset_path}'. Skipping preprocessing.")

    # Use corrected data as test data   path_to_zip, set_type=0, only_signal=False, device=None
    val_data = PartNetDataset(new_dataset_path, set_type=1)
    test_data = PartNetDataset(new_dataset_path, set_type=2)

    # Start training runs
    test_accuracies, test_losses, sub_logging_dirs = [], [], ["bbox_approach", "deepview_approach"]
    for idx, zip_file in enumerate([old_dataset_path, new_dataset_path]):
        train_data = PartNetDataset(zip_file, set_type=0)
        adaptation_data = PartNetDataset(zip_file, set_type=0, only_signal=True)

        # Train, validate and test IMCNN
        for trial_idx in range(trials):
            hist = train_single_imcnn(
                None,
                saving_path=f"{logging_dir}/state_dict_imcnn_{trial_idx}",
                n_epochs=epochs,
                adapt_data=adaptation_data,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data
            )

            test_accuracies.append(hist["test_accuracy"][-1])
            test_losses.append(hist["test_loss"][-1])

    # Compute and store Mann-Whitney U test
    mwutest_file_name = f"{logging_dir}/mann_whitney_u_test.txt"
    if not os.path.exists(mwutest_file_name):
        os.mknod(mwutest_file_name)

    acc_test_statistic, acc_p_value = sp.stats.ranksums(x=test_accuracies[0], y=test_accuracies[1])
    loss_test_statistic, loss_p_value = sp.stats.ranksums(x=test_losses[0], y=test_losses[1])

    with open(mwutest_file_name, "w") as f:
        f.write("### MANN-WHITNEY-U TEST ###\n")
        f.write(f"Test accuracy statistic: {acc_test_statistic}\n")
        f.write(f"Test accuracy p-value: {acc_p_value}\n")
        f.write(f"Test loss statistic: {loss_test_statistic}\n")
        f.write(f"Test loss p-value: {loss_p_value}\n")
        f.write("###########################\n")
        f.write(f"Captured test accuracies (bbox-approach): {test_accuracies[0]}\n")
        f.write(f"Captured test accuracies (deepview-approach): {test_accuracies[1]}\n")
        f.write("###########################\n")
        f.write(f"Captured test loss (bbox-approach): {test_losses[0]}\n")
        f.write(f"Captured test loss (deepview-approach): {test_losses[1]}\n")
        f.write("###########################\n")
