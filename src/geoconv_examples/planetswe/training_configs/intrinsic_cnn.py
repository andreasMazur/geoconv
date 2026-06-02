from geoconv_examples.planetswe.architectures.intrinsic_cnn import define_model
from geoconv_examples.planetswe.training import training


def start_training_run(bc_path,
                       swe_path,
                       template_radius,
                       kernel,
                       save_path,
                       template_resolutions=None,
                       learning_rate=0.001,
                       lr_decay_rate=1.0):
    if template_resolutions is None:
        template_resolutions = [(4, 8)]

    for (n_radial, n_angular) in template_resolutions:
        # Define model
        model = define_model(
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=template_radius,
            kernel=kernel,
            output_dims=[32, 64, 96, 64],
            learning_rate=learning_rate,
            lr_decay_rate=lr_decay_rate
        )
        training(
            model=model,
            bc_path=bc_path,
            swe_path=swe_path,
            return_rotations=False,
            save_path=f"{save_path}_{n_radial}_{n_angular}",
            epochs=10,
            random_seed=42,
            tensorboard_cb=False
        )
