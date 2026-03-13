from geoconv_examples.faust.architectures.hsn_cnn import define_model
from geoconv_examples.faust.training import training
from geoconv_examples.faust.training_configs.dictionaries import FAUST


def start_training_run(faust_path,
                       preprocessing,
                       save_path,
                       template_resolutions=None,
                       learning_rate=0.01,
                       lr_decay_rate=1.0):
    if template_resolutions is None:
        template_resolutions = [(4, 8)]
    for (n_radial, n_angular) in template_resolutions:
        for gpc_system_radius in [0.01, 0.02, 0.03, 0.04, 0.05]:
            # Define model
            template_radius = FAUST[(n_radial, n_angular)][preprocessing][gpc_system_radius]
            model = define_model(
                output_dims=[32, 64, 96, 64],
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
