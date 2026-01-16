from geoconv_examples.mnist.dataset import dataset

import keras_tuner as kt
import json


def hypertuning(mnist_atlas, n_radial, n_angular, batch_size, save_path, epochs=5):
    # Get data
    train_data, template_radius = dataset(
        mnist_atlas, set_type="train", n_radial=n_radial, n_angular=n_angular, batch_size=batch_size
    )
    test_data, _ = dataset(
        mnist_atlas, set_type="test", n_radial=n_radial, n_angular=n_angular, batch_size=batch_size
    )

    # Define hypermodel function
    def get_hypermodel(hp):
        model = define_model(
            output_dims=[
                hp.Int("units", min_value=8, max_value=64, step=2) for _ in range(hp.Int("n_layers", min_value=4, max_value=16))
            ],
            template_radius=template_radius,
            n_radial=n_radial,
            n_angular=n_angular,
            learning_rate=hp.Float("learning_rate", min_value=1e-6, max_value=0.1),
        )
        model.summary()
        return model

    # Hyperparameter search
    tuner = kt.BayesianOptimization(
        hypermodel=get_hypermodel,
        objective=kt.Objective("val_sparse_categorical_accuracy", direction="max"),
        max_trials=10_000,
        num_initial_points=10,
        seed=42
    )
    tuner.search(train_data, epochs=epochs, validation_data=test_data)

    # Save best model
    best_model = tuner.get_best_models()[0]
    if save_path[-6:] != ".keras":
        save_path += ".keras"
    best_model.save(save_path)


def training(mnist_atlas, n_radial, n_angular, batch_size, model, save_path, epochs=10):
    # Get data
    train_data, template_radius = dataset(
        mnist_atlas, set_type="train", n_radial=n_radial, n_angular=n_angular, batch_size=batch_size
    )
    test_data, _ = dataset(
        mnist_atlas, set_type="test", n_radial=n_radial, n_angular=n_angular, batch_size=batch_size
    )

    # Show model summary
    model.summary()

    # Train model
    history = model.fit(x=train_data, batch_size=batch_size, epochs=epochs, validation_data=test_data)

    # Save model
    if save_path[-6:] != ".keras":
        save_path += ".keras"
    model.save(save_path)

    # Save history
    with open(f"{save_path[:-6]}_history.json", "w") as f:
        json.dump(history.history, f, indent=4)
