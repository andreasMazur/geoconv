from geoconv_examples.faust.architectures.eman_cnn import define_model as eman
from geoconv_examples.faust.architectures.eman_p_cnn import define_model as eman_p
from geoconv_examples.faust.architectures.gem_cnn import define_model as gem_cnn
from geoconv_examples.faust.architectures.gem_p_cnn import define_model as gem_p_cnn
from geoconv_examples.faust.training import training
from geoconv_examples.faust.training_configs.dictionaries import FAUST


def start_training_run(faust_path,
                       preprocessing,
                       save_path,
                       model_type,
                       template_resolutions=None,
                       learning_rate=0.0004,
                       lr_decay_rate=1.0):
    """Starts the training run for a Harmonic surface network.

    Parameters
    ----------
    faust_path: str
        The path to the preprocessed FAUST dataset.
    preprocessing: str
        The used charting algorithm.
    save_path: str
        A path for the trained model can be saved.
    model_type: str
        A string describing which model shall be trained. Has to be one out of the following:
        [gem_cnn, eman, gem_p_cnn, eman_p].
    template_resolutions: list
        A list of tuples, each describing a template resolution.
    learning_rate: float
        The learning rate for the HSN model.
    lr_decay_rate: float
        The learning rate decay rate for the HSN model.
    """
    assert model_type in ["gem_cnn", "eman", "gem_p_cnn", "eman_p"], (
        "Select model type from: ['gem_cnn', 'eman', 'gem_p_cnn', 'eman_p']"
    )
    if model_type == "gem_cnn":
        define_model = gem_cnn
    elif model_type == "eman":
        define_model = eman
    elif model_type == "gem_p_cnn":
        define_model = gem_p_cnn
    else:
        define_model = eman_p

    if template_resolutions is None:
        template_resolutions = [(4, 8)]

    for (n_radial, n_angular) in template_resolutions:
        for gpc_system_radius in [0.01, 0.02, 0.03, 0.04, 0.05]:
            # Define model
            template_radius = FAUST[(n_radial, n_angular)][preprocessing][gpc_system_radius]
            model = define_model(
                input_types=[
                    [0 for _ in range(int(n_radial * n_angular * 3))],
                    [x for _ in range(8) for x in [1, 2]],
                    [x for _ in range(16) for x in [1, 2]],
                    [x for _ in range(24) for x in [1, 2]],
                ],
                output_types=[
                    [x for _ in range(8) for x in [1, 2]],
                    [x for _ in range(16) for x in [1, 2]],
                    [x for _ in range(24) for x in [1, 2]],
                    [x for _ in range(16) for x in [1, 2]]
                ],
                preprocess_method=preprocessing,
                gpc_radius=gpc_system_radius,
                template_radius=template_radius,
                n_radial=n_radial,
                n_angular=n_angular,
                learning_rate=learning_rate,
                lr_decay_rate=lr_decay_rate,
                faust_path=faust_path,
            )

            # Start training
            training_run_save_path = f"{save_path}/{n_radial}_{n_angular}_{'_'.join(f'{gpc_system_radius}'.split('.'))}_{template_radius}"
            print(f"Currently running: {training_run_save_path}")
            training(
                faust_path=faust_path,
                n_radial=n_radial,
                n_angular=n_angular,
                preprocess_method=preprocessing,
                gpc_radius=gpc_system_radius,
                template_radius=template_radius,
                model=model,
                save_path=training_run_save_path,
                epochs=200,
                return_rotations=True
            )
