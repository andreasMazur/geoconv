from geoconv_examples.mpi_faust.tensorflow.training_demo import train_model
from geoconv_examples.mpi_faust_segmentation.tensorflow.convert_dataset import convert_dataset

from pathlib import Path


def run_experiment(registration_path, old_dataset_path, new_dataset_path):
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
        ### PRE-PROCESSING ###
        n_radial=5,
        n_angular=8,
        registration_path=registration_path,
        # Make sure you have installed: https://github.com/uhlmanngroup/pyshot (do not use `pip install pyshot`!)
        compute_shot=True,
        precomputed_gpc_radius=R,
        ### GENERAL ###
        template_radius=R * 0.75,
        logging_dir="./tensorflow_seg_demo_logging",
        processes=1,
        layer_conf=[(96, 1)],
        init_lr=0.00165,
        weight_decay=0.005,
        model="dirac",
        segmentation=True,
        epochs=10,
        seeds=[10]
    )

    # 2.) Run experiment with corrected segmentation labels
