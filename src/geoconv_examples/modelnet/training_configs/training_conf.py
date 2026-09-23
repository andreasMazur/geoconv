from geoconv_examples.modelnet.architectures.eman_cnn import define_model as eman
from geoconv_examples.modelnet.architectures.eman_p_cnn import define_model as eman_p
from geoconv_examples.modelnet.architectures.gem_cnn import define_model as gem_cnn
from geoconv_examples.modelnet.architectures.gem_p_cnn import define_model as gem_p_cnn
from geoconv_examples.modelnet.training import training
from geoconv_examples.modelnet.training_configs.dictionaries import MN10


def start_training_run(mn10_path,
                       preprocessing,
                       save_path,
                       model_type,
                       n_radial,
                       n_angular,
                       chart_max_radius,
                       learning_rate=0.0004,
                       lr_decay_rate=1.0):
    """Starts the training run for a equivariant mesh convolution architectures.

    Parameters
    ----------
    mn10_path: str
        The path to the preprocessed ModelNet10 dataset.
    preprocessing: str
        The used charting algorithm.
    save_path: str
        The path which points to where the model and benchmark statistics will be saved.
    model_type: str
        The gauge-equivariant architecture type. Must be one from: ['gem_cnn', 'eman', 'gem_p_cnn', 'eman_p'].
    n_radial: int
        The amount of radial coordinates considered by the discretized template.
    n_angular: int
        The amount of angular coordinates considered by the discretized template.
    chart_max_radius: float
        The maximum allowed chart radius.
    learning_rate: float
        The learning rate
    lr_decay_rate: float
        The decay rate for the learning rate
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

    # Define model
    template_radius = MN10[(n_radial, n_angular)][preprocessing][chart_max_radius]
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
            [0 for _ in range(64)]  # causes 128 dimensions but every second element is zero => 64 dims
        ],
        template_radius=template_radius,
        n_radial=n_radial,
        n_angular=n_angular,
        learning_rate=learning_rate,
        lr_decay_rate=lr_decay_rate
    )

    # Start training
    training_run_save_path = f"{save_path}/{n_radial}_{n_angular}_{'_'.join(f'{chart_max_radius}'.split('.'))}_{template_radius}"
    print(f"Currently running: {training_run_save_path}")
    training(
        mn10_path=mn10_path,
        n_radial=n_radial,
        n_angular=n_angular,
        preprocess_method=preprocessing,
        gpc_radius=chart_max_radius,
        model=model,
        save_path=training_run_save_path,
        epochs=200,
        random_seed=42,
        return_rotations=True
    )
