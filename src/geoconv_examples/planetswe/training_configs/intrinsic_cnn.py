from geoconv_examples.planetswe.architectures.intrinsic_cnn import define_model
from geoconv_examples.planetswe.training import training
from geoconv_examples.planetswe.training_configs.dictionaries import PLANETSWE


def start_training_run(bc_path,
                       swe_path,
                       max_chart_radius,
                       charting_method,
                       kernel,
                       save_path,
                       template_resolutions=None,
                       learning_rate=0.001,
                       lr_decay_rate=1.0,
                       tensorboard_cb=False,
                       batch_size=1,
                       predict_residual=False):
    if template_resolutions is None:
        template_resolutions = [(4, 8)]

    for (n_radial, n_angular) in template_resolutions:
        # Retrieve template radius
        template_radius = PLANETSWE[(n_radial, n_angular)][charting_method][max_chart_radius]

        # Define model
        model = define_model(
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=template_radius,
            kernel=kernel,
            output_dims=[32],
            learning_rate=learning_rate,
            lr_decay_rate=lr_decay_rate,
            predict_residual=predict_residual
        )

        # Start training
        training(
            model=model,
            bc_path=bc_path,
            swe_path=swe_path,
            return_rotations=False,
            save_path=f"{save_path}_{n_radial}_{n_angular}",
            random_seed=42,
            tensorboard_cb=tensorboard_cb,
            batch_size=batch_size,
            add_input_zero_dim=False,
            predict_residual=predict_residual
        )
