from geoconv_examples.mpi_faust.pytorch.faust_data_set import FaustDataset
from geoconv_examples.mpi_faust.pytorch.model import Imcnn
from geoconv_examples.mpi_faust.data.preprocess_faust import preprocess_faust
from geoconv.utils.measures import princeton_benchmark

from pathlib import Path
from torch import nn

import torch
import numpy as np
import sys
import json
import os


def train_model(reference_mesh_path,
                signal_dim,
                preprocessed_data,
                n_radial=5,
                n_angular=8,
                registration_path="",
                compute_shot=True,
                geodesic_diameters_path="",
                precomputed_gpc_radius=0.037,
                template_radius=0.027744965069279016,
                logging_dir="./imcnn_training_logs",
                processes=1,
                init_lr=0.00165,
                weight_decay=0.005,
                layer_conf=None,
                model="dirac",
                add_noise=False,
                reference_mesh_diameter=2.2093810817030244,
                early_stop=20):
    """Trains one singular IMCNN

    Parameters
    ----------
    reference_mesh_path: str
        The path to the reference mesh file.
    signal_dim: int
        The dimensionality of the mesh signal
    preprocessed_data: str
        The path to the pre-processed data. If you have not pre-processed your data so far and saved it under the given
        path, this script will execute pre-processing for you. For this to work, you need to pass the arguments which
        are annotated with '[REQUIRED FOR PRE-PROCESSING]'. If pre-processing is not required, you can ignore those
        arguments.
    reference_mesh_diameter: float
        [REQUIRED FOR BENCHMARKING] The geodesic diameter of the reference mesh. Defaults to the diameter of the first
        registration mesh (tr_reg_000.ply) in the training set of the FAUST data set. If other reference mesh is
        selected, adjust this parameter accordingly!
    n_radial: int
        [REQUIRED FOR PRE-PROCESSING] The amount of radial coordinates for the template.
    n_angular: int
        [REQUIRED FOR PRE-PROCESSING] The amount of angular coordinates for the template.
    registration_path: str
        [REQUIRED FOR PRE-PROCESSING] The path of the training-registration files in the FAUST data set.
    compute_shot: bool
        [REQUIRED FOR PRE-PROCESSING] Whether to compute SHOT-descriptors during preprocessing as the mesh signal
    geodesic_diameters_path: str
        [REQUIRED FOR PRE-PROCESSING] The path to pre-computed geodesic diameters for the FAUST-registration meshes.
    precomputed_gpc_radius: float
        [REQUIRED FOR PRE-PROCESSING] The GPC-system radius to use for GPC-system computation. If not provided, the
        script will calculate it.
    template_radius: float
        [OPTIONAL] The template radius of the ISC-layer (the one used during preprocessing, defaults to radius for FAUST
        data set).
    logging_dir: str
        [OPTIONAL] The path to the folder where logs will be stored
    processes: int
        [OPTIONAL] The amount of concurrent processes. Affects preprocessing and Princeton benchmark.
    init_lr: float
        [OPTIONAL] Initial learning rate.
    weight_decay: float
        [OPTIONAL] Weight decay.
    layer_conf: list
        [OPTIONAL] List of tuples: The first entry references the output dimensions of the i-th ISC-layer, The second
        entry references of skips between each rotation while computing the convolution (rotation delta).
    model: str
        [OPTIONAL] Which model variant (['dirac', 'geodesic', 'zero']) shall be tuned.
    add_noise: bool
        [OPTIONAL] Adds Gaussian noise to the mesh data.
    early_stop: int
        [OPTIONAL] The amount of epochs for early stopping.
    """
    # Create logging dir
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Load data
    preprocess_zip = f"{preprocessed_data}.zip"
    if not Path(preprocess_zip).is_file():
        template_radius = preprocess_faust(
            n_radial=n_radial,
            n_angular=n_angular,
            target_dir=preprocess_zip[:-4],
            registration_path=registration_path,
            shot=compute_shot,
            geodesic_diameters_path=geodesic_diameters_path,
            precomputed_gpc_radius=precomputed_gpc_radius,
            processes=processes,
            add_noise=add_noise
        )
    else:
        print(f"Found preprocess-results: '{preprocess_zip}'. Skipping preprocessing.")

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = [10, 20, 30, 40, 50]
    for exp_number in range(len(seeds)):
        # Set seeds
        torch.manual_seed(seeds[exp_number])
        np.random.seed(seeds[exp_number])
        sys.stdout.write(f"\n### Experiment no. {exp_number}")

        # Define model
        imcnn = Imcnn(
            signal_dim=signal_dim,
            kernel_size=(n_radial, n_angular),
            template_radius=template_radius,
            adapt_data=FaustDataset(preprocess_zip, set_type=0, only_signal=True, device=device),
            layer_conf=layer_conf,
            variant=model
        )
        imcnn.to(device)
        imcnn = torch.compile(imcnn)

        # Define loss, optimizer and scheduler
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(
            params=imcnn.parameters(), lr=init_lr, weight_decay=weight_decay, eps=1e-07
        )
        opt.param_groups[0]["initial_lr"] = init_lr

        # Fit model
        training_history = {}
        best_loss = np.inf
        stale_counter = 0
        for epoch in range(200):
            sys.stdout.write("\n")  # pretty printing

            # Training
            train_data = FaustDataset(preprocess_zip, set_type=0, device=device)
            train_dict = imcnn.train_loop(
                train_data,
                loss_fn,
                opt,
                decay_rate=0.99,
                decay_steps=500,
                verbose=True,
                epoch=epoch,
                prev_steps=epoch * 70
            )

            # Validation
            val_data = FaustDataset(preprocess_zip, set_type=1, device=device)
            val_dict = imcnn.validation_loop(val_data, loss_fn, verbose=True)
            val_loss = val_dict["val_epoch_loss"].item()

            # Remember epoch statistics
            training_history[f"epoch_{epoch}"] = {
                "epoch": epoch,
                "loss": train_dict["epoch_loss"].item(),
                "Accuracy": train_dict["epoch_accuracy"].item(),
                "Validation Loss": val_loss,
                "Validation Accuracy": val_dict["val_epoch_accuracy"].item()
            }

            # Log best model
            if val_loss < best_loss:
                best_loss = val_loss
                imcnn_path = f"{logging_dir}/imcnn_exp_{exp_number}_epoch_{epoch}"
                torch.save(imcnn.state_dict(), imcnn_path)
                stale_counter = 0
            else:
                stale_counter += 1

            # Early stopping
            if stale_counter >= early_stop:
                sys.stdout.write("\nEarly stopping.")
                break

        print("\n")  # pretty printing

        # Log training statistics
        with open(f"{logging_dir}/training_history_{exp_number}.json", "w") as file:
            json.dump(training_history, file, indent=4)

        test_dataset = FaustDataset(preprocess_zip, set_type=2, device=device)
        with torch.no_grad():
            princeton_benchmark(
                imcnn=imcnn,
                test_dataset=test_dataset,
                ref_mesh_path=reference_mesh_path,
                file_name=f"{logging_dir}/model_benchmark_{exp_number}",
                processes=processes,
                geodesic_diameter=reference_mesh_diameter,
                pytorch_model=True
            )
