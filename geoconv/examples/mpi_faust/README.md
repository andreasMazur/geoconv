# Example script for a pre-processing, hyperparameter tuning and training pipeline

Be aware that the demo script can compute for a while. We strongly recommend to run the script as a compute job
on a GPU-cluster. For a quick insight into intermediate pre-processing results, please take a look into the
Stanford-bunny example.

You can call the demo, by writing a script that calls `training_demo`. E.g:
```python
from geoconv.examples.mpi_faust.training_demo import training_demo

if __name__ == "__main__":
    rp = "/home/user/geoconv/geoconv/examples/mpi_faust/data/MPI-FAUST/training/registrations"
    R = 0.036993286759038686
    training_demo(
        preprocess_target_dir="./test_training_demo",
        registration_path=rp,
        log_dir="./logs_training_demo",
        reference_mesh_path=f"{rp}/tr_reg_000.ply",
        amt_convolutions=1,
        imcnn_variant="dirac_lite",
        tuner_variant="hyperband",
        amt_splits=10,
        n_radial=5,
        n_angular=8,
        compute_shot=True,  # Make sure you have installed: https://github.com/uhlmanngroup/pyshot (do not use `pip install pyshot`!)
        signal_dim=544,  # Set it to 3 if `compute_shot=False`
        geodesic_diameters_path="/home/user/geoconv/geoconv/examples/mpi_faust/geodesic_diameters.npy",
        precomputed_gpc_radius=R,
        kernel_radius=R * 0.75,
        save_gpc_systems=False  # Set this to 'True' in case you want to inspect GPC-systems
    )
```

## Installing pyshot

If you want `training_demo` to compute SHOT-descriptors, you need to install `pyshot` into your environment from:
https://github.com/uhlmanngroup/pyshot

This repository raises additional dependencies, which you can install with the following command:
```bash
    sudo apt install libflann-dev libeigen3-dev lz4
```