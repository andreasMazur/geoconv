# Example script for a pre-processing, hyperparameter tuning and training pipeline

Be aware that the demo script can compute for a while. We strongly recommend to run the script as a compute job
on a GPU-cluster. For a quick insight into intermediate pre-processing results, please take a look into the
Stanford-bunny example.

For this script to run, you will need to download the FAUST dataset:
https://faust-leaderboard.is.tuebingen.mpg.de/

You can call the demo, by writing a script that calls `training_demo`. E.g:

```python
from geoconv_examples.mpi_faust.tensorflow.training_demo import train_model

if __name__ == "__main__":
    rp = "/home/user/src/src/geoconv_examples/mpi_faust/data/MPI-FAUST/training/registrations"
    R = 0.036993286759038686
    train_model(
        reference_mesh_path=f"{rp}/tr_reg_000.ply",
        signal_dim=544,  # Set it to 3 if `compute_shot=False`
        preprocessed_data="/home/user/src/src/geoconv_examples/mpi_faust/data/preprocessed_dataset_5_8",
        ### PRE-PROCESSING ###
        n_radial=5,
        n_angular=8,
        registration_path=rp,
        compute_shot=True,
        # Make sure you have installed: https://github.com/uhlmanngroup/pyshot (do not use `pip install pyshot`!)
        geodesic_diameters_path="/home/user/src/src/geoconv_examples/mpi_faust/data/geodesic_diameters.npy",
        precomputed_gpc_radius=R,
        ### GENERAL ###
        template_radius=R * 0.75,
        logging_dir="./tensorflow_demo_logging",
        processes=1,
        layer_conf=[(96, 1), (256, 1), (384, 1), (384, 1)],
        init_lr=0.00165,
        weight_decay=0.005,
        model="dirac"
    )
```

## Installing pyshot

If you want `training_demo` to compute SHOT-descriptors, you need to install `pyshot` into your environment from:
https://github.com/uhlmanngroup/pyshot

This repository raises additional dependencies, which you can install with the following command:
```bash
    sudo apt install libflann-dev libeigen3-dev lz4
```