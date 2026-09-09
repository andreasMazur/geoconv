from geoconv_examples.planetswe.architectures.gem_cnn import define_model as gem_cnn
from geoconv_examples.planetswe.architectures.gem_p_cnn import define_model as gem_p_cnn
from geoconv_examples.planetswe.architectures.eman_cnn import define_model as eman_cnn
from geoconv_examples.planetswe.architectures.eman_p_cnn import define_model as eman_p_cnn
from geoconv_examples.planetswe.training_configs.dictionaries import PLANETSWE
from geoconv_examples.planetswe.training import training


def start_training_run(bc_path,
                       swe_path,
                       max_chart_radius,
                       charting_method,
                       model_type,
                       save_path,
                       template_resolutions=None,
                       learning_rate=0.001,
                       lr_decay_rate=1.0,
                       tensorboard_cb=False,
                       batch_size=1,
                       predict_residual=False):
    """Starts training runs for the selected architecture.

    Parameters
    ----------
    bc_path: str
        The path that points to the location where the barycentric coordinates for the sphere are saved.
    swe_path: str
        The path that points to the downloaded PlanetSWE dataset. Used to retrieve sphere signals.
    max_chart_radius: float
        The max chart radius.
    charting_method: str
        The charting method.
    model_type: str
        The model type. Has to be selected from: ['gem_cnn', 'gem_p_cnn', 'eman', 'eman_p']. Defaults to 'eman_p'.
    save_path: str
        The path that points to the location where the logging directory should be created, saving model and logs.
    template_resolutions: list
        A list of template resolutions, i.e., [(n_radial, n_angular), ...].
    learning_rate: float
        The learning rate.
    lr_decay_rate: float
        The lr decay rate.
    tensorboard_cb: bool
        Whether to enable the tensorboard callback.
    batch_size: int
        The batch size.
    predict_residual: bool
        Whether the model should predict residuals which are added onto the current state instead of the full state for
        the next time step.
    """
    assert model_type in ["gem_cnn", "eman", "gem_p_cnn", "eman_p"], (
        "Select model type from: ['gem_cnn', 'eman', 'gem_p_cnn', 'eman_p']"
    )
    if model_type == "gem_cnn":
        define_model = gem_cnn
    elif model_type == "eman":
        define_model = eman_cnn
    elif model_type == "gem_p_cnn":
        define_model = gem_p_cnn
    else:
        define_model = eman_p_cnn

    if template_resolutions is None:
        template_resolutions = [(4, 8)]

    for (n_radial, n_angular) in template_resolutions:
        # Retrieve template radius
        template_radius = PLANETSWE[(n_radial, n_angular)][charting_method][max_chart_radius]

        # Define model
        model = define_model(
            input_types=[
                [1, 0]  # [velocity, height]
            ],
            output_types=[
                [x for _ in range(8) for x in [1, 2]]
            ],
            template_radius=template_radius,
            n_radial=n_radial,
            n_angular=n_angular,
            learning_rate=learning_rate,
            lr_decay_rate=lr_decay_rate,
            predict_residual=predict_residual
        )

        # Start training
        training(
            model=model,
            bc_path=bc_path,
            swe_path=swe_path,
            return_rotations=True,
            save_path=f"{save_path}_{n_radial}_{n_angular}",
            random_seed=42,
            tensorboard_cb=tensorboard_cb,
            batch_size=batch_size,
            add_input_zero_dim=True,
            predict_residual=predict_residual
        )
