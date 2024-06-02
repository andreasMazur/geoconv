from geoconv.utils.common import read_template_configurations
from geoconv_examples.modelnet_40.dataset import load_preprocessed_modelnet, MODELNET40_FOLDS

import os


def training(bc_path, logging_dir):
    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Prepare template configurations
    template_configurations = read_template_configurations(bc_path)

    # Run experiments
    for (n_radial, n_angular, template_radius) in template_configurations:
        for exp_no in range(len(MODELNET40_FOLDS.keys())):
            # Load data
            train_data = load_preprocessed_modelnet(
                bc_path, n_radial, n_angular, template_radius, is_train=True, split=exp_no
            )
            test_data = load_preprocessed_modelnet(
                bc_path, n_radial, n_angular, template_radius, is_train=False, split=exp_no
            )
