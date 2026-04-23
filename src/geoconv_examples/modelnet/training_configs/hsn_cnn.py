from geoconv_examples.modelnet.architectures.hsn_cnn import define_model
from geoconv_examples.modelnet.training import training
from geoconv_examples.modelnet.training_configs.dictionaries import MN10


def start_training_run(mn10_path,
                       preprocessing,
                       save_path,
                       n_radial,
                       n_angular,
                       learning_rate=0.001,
                       lr_decay_rate=1.0):
    for gpc_system_radius in [0.03, 0.04, 0.05, 0.06, 0.07]:
        # Define model
        template_radius = MN10[(n_radial, n_angular)][preprocessing][gpc_system_radius]
        model = define_model(
            output_dims=[32, 64, 96, 64],
            template_radius=template_radius,
            n_radial=n_radial,
            n_angular=n_angular,
            learning_rate=learning_rate,
            lr_decay_rate=lr_decay_rate,
        )

        # Start training
        training_run_save_path = f"{save_path}/{n_radial}_{n_angular}_{'_'.join(f'{gpc_system_radius}'.split('.'))}_{template_radius}"
        print(f"Currently running: {training_run_save_path}")
        training(
            mn10_path=mn10_path,
            n_radial=n_radial,
            n_angular=n_angular,
            preprocess_method=preprocessing,
            gpc_radius=gpc_system_radius,
            model=model,
            save_path=training_run_save_path,
            epochs=200,
            random_seed=42,
            return_rotations=True
        )
