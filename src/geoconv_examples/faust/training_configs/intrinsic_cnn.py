from geoconv_examples.faust.architectures.intrinsic_cnn import define_model
from geoconv_examples.faust.training import training
from geoconv_examples.faust.training_configs.dictionaries import FAUST


def start_training_run(faust_path, preprocessing, kernel, save_path):
    if kernel == "dirac":
        learning_rate = 0.009
    elif kernel == "geodesic":
        learning_rate = 0.007
    else:
        raise ValueError("kernel must be either 'dirac' or 'geodesic'.")

    for (n_radial, n_angular) in [(2, 4), (4, 8)]:
        for gpc_system_radius in [0.05, 0.1, 0.15, 0.2]:
            for template_radius in ["min", "median", "max"]:
                # Define model
                radius = FAUST[preprocessing][gpc_system_radius][template_radius]
                model = define_model(
                    output_dims=[32, 64, 96, 64],
                    template_radius=radius,
                    n_radial=n_radial,
                    n_angular=n_angular,
                    kernel=kernel,
                    learning_rate=learning_rate,
                    faust_path=faust_path,
                )

                # Start training
                gpc_system_radius = "_".join(f"{gpc_system_radius}".split("."))
                training_run_save_path = f"{save_path}/{n_radial}_{n_angular}_{gpc_system_radius}_{template_radius}"
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
