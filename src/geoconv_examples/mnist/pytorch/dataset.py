from geoconv.utils.data_generator import barycentric_coordinates_generator

from torchvision import datasets
from torch.utils.data import DataLoader

import numpy as np


def load_preprocessed_mnist(dataset_path,
                            n_radial,
                            n_angular,
                            template_radius,
                            set_type,
                            batch_size=8,
                            mnist_folder=None):
    """Adds barycentric coordinates to the MNIST dataset and reshapes images to vectors.

    Parameters
    ----------
    dataset_path: str
        The path to the preprocessed dataset.
    n_radial: int
        The amount of radial coordinates used during BC-computation.
    n_angular: int
        The amount of angular coordinates used during BC-computation.
    template_radius: float
        The considered template radius during BC-computation.
    set_type: str
        The set type. Either 'train' or 'test'.
    batch_size: int
        The batch-size.
    mnist_folder: str | None
        The path to the folder containing the MNIST dataset. If not stored there, the dataset will be downloaded.

    Returns
    -------
    torch.utils.data.DataLoader:
        A dataset containing MNIST-images and labels together with barycentric coordinates to train an IMCNN.
    """
    if mnist_folder is None:
        mnist_folder = "./data"

    # Load barycentric coordinates
    barycentric_coordinates = barycentric_coordinates_generator(
        dataset_path, n_radial, n_angular, template_radius, batch_size=1, return_filename=False
    )

    # Add barycentric coordinates to MNIST data
    for bc in barycentric_coordinates:
        def make_compatible(image):
            image = np.array(image).reshape((-1, 1))
            return image / 255., bc
        dataloader = DataLoader(
            datasets.MNIST(mnist_folder, train=set_type == "train", download=True, transform=make_compatible),
            batch_size=batch_size,
            shuffle=True
        )
        return dataloader
