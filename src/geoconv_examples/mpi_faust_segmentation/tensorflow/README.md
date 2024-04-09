# Example: FAUST segmentation

```python
from geoconv_examples.mpi_faust.tensorflow.training_demo import train_model
from geoconv_examples.mpi_faust_segmentation.data.segment_meshes import compute_seg_labels
from geoconv_examples.mpi_faust_segmentation.tensorflow.convert_dataset import sc_to_seg_converter


if __name__ == "__main__":
    ##################
    # Configure paths
    ##################
    registration_path = "/home/user/datasets/MPI-FAUST/training/registrations"
    labels_path = "./segmentation_labels"  # Where shall the segmentation labels be stored?
    old_dataset_path = "/home/user/datasets/preprocessed_faust_5_8.zip"  # Where is the preprocessed FAUST dataset?
    new_dataset_path = "/home/user/datasets/preprocessed_faust_5_8_seg"  # Where shall the dataset for segmenting be stored?
    ##################
    
    compute_seg_labels(registration_path, labels_path, verbose=False)
    sc_to_seg_converter(
        dataset_path=old_dataset_path,
        new_dataset_path=new_dataset_path,
        segmentation_labels_path=f"{labels_path}/segmentation_labels.npy"
    )

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
        layer_conf=[(96, 1), (256, 1), (384, 1), (384, 1)],
        init_lr=0.00165,
        weight_decay=0.005,
        model="dirac",
        segmentation=True
    )
```