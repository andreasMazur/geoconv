from geoconv_examples.mpi_faust.tensorflow.training_demo import train_model
from geoconv_examples.mpi_faust_segmentation.tensorflow.convert_dataset import convert_dataset

from pathlib import Path
from matplotlib import pyplot as plt

import pandas as pd


def visualize_csv(csv_path, figure_name="training_statistics"):
    csv = pd.read_csv(csv_path)

    # Configure plot
    n_rows, n_cols, cm = 2, 1, 1/2.54
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12*cm, 12.2*cm), sharex=True)
    for j in range(n_rows):
        axs[j].grid()

    # Configure axes
    axs[0].plot(csv["loss"], label="loss")
    axs[0].plot(csv["val_loss"], label="val_loss")
    axs[0].set_ylabel("Loss")

    axs[1].plot(csv["sparse_categorical_accuracy"])
    axs[1].plot(csv["val_sparse_categorical_accuracy"])
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Sparse Categorical Accuracy")

    plt.savefig(f"{figure_name}.svg", bbox_inches="tight")
    plt.show()


def run_experiment(registration_path, old_dataset_path, new_dataset_path, logging_dir):
    #######################################################
    # 1.) Run experiment w/o corrected segmentation labels
    #######################################################
    # Convert shape correspondence labels to segmentation labels
    converted_zip = f"{new_dataset_path}.zip"
    if not Path(converted_zip).is_file():
        convert_dataset(
            registration_path=registration_path,
            old_dataset_path=old_dataset_path,
            new_dataset_path=new_dataset_path
        )
    else:
        print(f"Found converted dataset file: '{converted_zip}'. Skipping preprocessing.")

    R = 0.036993286759038686
    train_model(
        reference_mesh_path=f"{registration_path}/tr_reg_000.ply",
        preprocessed_data=new_dataset_path,
        n_radial=5,
        n_angular=8,
        registration_path=registration_path,
        compute_shot=True,
        precomputed_gpc_radius=R,
        template_radius=R * 0.75,
        logging_dir=logging_dir,
        processes=1,
        layer_conf=[(96, 1)],
        init_lr=0.00165,
        weight_decay=0.005,
        model="dirac",
        segmentation=True,
        epochs=10,
        seeds=[10]
    )

    # Visualize training results
    visualize_csv(f"{logging_dir}/training_0.log", figure_name=f"{logging_dir}/training_statistics")

    ########################################################
    # 2.) Run experiment with corrected segmentation labels
    ########################################################
    # Load corrected segmentation labels
