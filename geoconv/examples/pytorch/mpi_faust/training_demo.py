from geoconv.examples.pytorch.mpi_faust.faust_data_set import FaustDataset
from geoconv.examples.tensorflow.mpi_faust.faust_data_set import load_preprocessed_faust
from geoconv.examples.pytorch.mpi_faust.model import Imcnn
from geoconv.examples.tensorflow.mpi_faust.preprocess_faust import preprocess_faust
from geoconv.utils.measures import princeton_benchmark

from pathlib import Path
from torch import nn

import torch
import numpy as np


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
                reference_mesh_diameter=2.2093810817030244):
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
    """
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = [10, 20, 30, 40, 50]
    for exp_number in range(len(seeds)):
        # Set seeds
        torch.manual_seed(seeds[exp_number])
        np.random.seed(seeds[exp_number])

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

        # Define loss, optimizer and scheduler
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(params=imcnn.parameters(), lr=init_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.95)

        # TODO: Reduce memory consumption
        # Fit model
        for step, ((signal, bc), gt) in enumerate(FaustDataset(preprocess_zip, set_type=0, device=device)):
            opt.zero_grad()
            pred = imcnn([signal, bc])
            loss = loss_fn(pred, gt)
            loss.backward()
            opt.step()
            if step % 500 == 0:
                scheduler.step()

        # TODO: validation; FaustDataset(preprocess_zip, set_type=1, device=device)

        # TODO: Define callbacks
        # csv = keras.callbacks.CSVLogger(f"{logging_dir}/training_{exp_number}.log")
        # stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
        # tb = keras.callbacks.TensorBoard(
        #     log_dir=f"{logging_dir}/tensorboard_{exp_number}",
        #     histogram_freq=1,
        #     write_graph=False,
        #     write_steps_per_second=True,
        #     update_freq="epoch",
        #     profile_batch=(1, 70)
        # )

        # TODO: Evaluate model with Princeton benchmark
        # test_dataset = FaustDataset(preprocess_zip, set_type=2, device=device)
        # princeton_benchmark(
        #     imcnn=imcnn,
        #     test_dataset=test_dataset,
        #     ref_mesh_path=reference_mesh_path,
        #     file_name=f"{logging_dir}/model_benchmark_{exp_number}",
        #     processes=processes,
        #     geodesic_diameter=reference_mesh_diameter
        # )
