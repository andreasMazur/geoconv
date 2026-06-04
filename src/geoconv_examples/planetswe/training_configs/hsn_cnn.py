from geoconv_examples.planetswe.architectures.hsn_cnn import define_model
from geoconv_examples.planetswe.training import training
from geoconv_examples.planetswe.training_configs.dictionaries import PLANETSWE


def start_training_run(bc_path,
                       swe_path,
                       max_chart_radius,
                       charting_method,
                       save_path,
                       template_resolutions=None,
                       learning_rate=0.001,
                       lr_decay_rate=1.0,
                       epochs=10,
                       tensorboard_cb=False,
                       batch_size=1,
                       rollout_t_max=100):
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
            output_dims=[32],
            learning_rate=learning_rate,
            lr_decay_rate=lr_decay_rate
        )

        # Start training
        training(
            model=model,
            bc_path=bc_path,
            swe_path=swe_path,
            return_rotations=True,
            save_path=f"{save_path}_{n_radial}_{n_angular}",
            epochs=epochs,
            random_seed=42,
            tensorboard_cb=tensorboard_cb,
            batch_size=batch_size,
            add_input_zero_dim=True,
            rollout_t_max=rollout_t_max
        )
