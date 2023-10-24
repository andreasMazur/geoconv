# Example script for a pre-processing, hyperparameter tuning and training pipeline

Be aware that the demo script can compute for a while. We strongly recommend to run the script as a compute job
on a GPU-cluster. For a quick insight into intermediate pre-processing results, please take a look into the
Stanford-bunny example.

You can call the demo, by writing a script that calls `training_demo`. E.g:

```python
from geoconv.examples.mpi_faust.training_pipeline_demo import training_pipeline_demo

if __name__ == "__main__":
    rp = "/home/user/geoconv/geoconv/examples/mpi_faust/data/MPI-FAUST/training/registrations"
    R = 0.036993286759038686
    training_pipeline_demo(
        reference_mesh_path=f"{rp}/tr_reg_000.ply",
        signal_dim=544,  # Set it to 3 if `compute_shot=False`
        preprocessed_data="./preprocessed_data",
        ### PRE-PROCESSING ###
        n_radial=5,
        n_angular=8,
        registration_path=rp,
        compute_shot=True,  # Make sure you have installed: https://github.com/uhlmanngroup/pyshot (do not use `pip install pyshot`!)
        geodesic_diameters_path="/home/user/geoconv/geoconv/examples/mpi_faust/geodesic_diameters.npy",
        precomputed_gpc_radius=0.037,
        save_gpc_systems=False,  # Set this to 'True' in case you want to inspect GPC-systems
        ### GENERAL ###
        template_radius=0.028,  # Should be smaller than 'precomputed_gpc_radius'
        logging_dir="./imcnn_training_logs",
        output_dim=128,
        splits=10,
        ### HYPERMODEL CONFIGURATION ###
        amt_convolutions=1,
        rotation_delta=1,
        imcnn_variant="dirac_lite",
        tuner_variant="hyperband"
    )
```

In case you want to skip hyperparameter tuning and train only one model, you can execute:
```python
from geoconv.examples.mpi_faust.train_one_imcnn import train_model

if __name__ == "__main__":
    rp = "/home/user/geoconv/geoconv/examples/mpi_faust/data/MPI-FAUST/training/registrations"
    R = 0.036993286759038686
    train_model(
        reference_mesh_path=f"{rp}/tr_reg_000.ply",
        signal_dim=544,  # Set it to 3 if `compute_shot=False`
        preprocessed_data="./preprocessed_data",
        ### PRE-PROCESSING ###
        n_radial=5,
        n_angular=8,
        registration_path=rp,
        compute_shot=True,  # Make sure you have installed: https://github.com/uhlmanngroup/pyshot (do not use `pip install pyshot`!)
        geodesic_diameters_path="/home/user/geoconv/geoconv/examples/mpi_faust/geodesic_diameters.npy",
        precomputed_gpc_radius=0.037,
        save_gpc_systems=False,  # Set this to 'True' in case you want to inspect GPC-systems
        ### GENERAL ###
        template_radius=0.028,  # Should be smaller than 'precomputed_gpc_radius'
        logging_dir="./imcnn_training_logs",
        output_dim=128,
        amt_templates=1,
        splits=10
    )
```

## Installing pyshot

If you want `training_demo` to compute SHOT-descriptors, you need to install `pyshot` into your environment from:
https://github.com/uhlmanngroup/pyshot

This repository raises additional dependencies, which you can install with the following command:
```bash
    sudo apt install libflann-dev libeigen3-dev lz4
```