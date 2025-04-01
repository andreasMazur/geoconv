from geoconv.utils.data_generator import barycentric_coordinates_generator

from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

import torch
import numpy as np


class ProcessedMNIST(Dataset):
    """A dataset containing formatted MNIST-images and labels together with barycentric coordinates.

    Attributes
    ----------
    images: torch.Tensor
        The images of MNIST, formatted into vectors.
    bc: torch.Tensor
        One set of barycentric coordinates that can be used for any images of MNIST.
    labels:
        The labels for the MNIST-images.
    """
    def __init__(self, images, bc, labels):
        self.images = images.reshape(images.shape[0], -1, 1) / 255.
        self.bc = torch.from_numpy(bc)
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return (self.images[idx], self.bc), self.labels[idx]


def load_preprocessed_mnist(dataset_path,
                            n_radial,
                            n_angular,
                            template_radius,
                            set_type,
                            batch_size=8,
                            mnist_folder=None,
                            indices=None):
    """Loads MNIST while adding barycentric coordinates to the dataset and reshapes images to vectors.

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
    indices: np.ndarray | None
        The indices if elements of the dataset to load.

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

        dataset = datasets.MNIST(mnist_folder, train=set_type == "train", download=True)
        if indices is None:
            dataset = ProcessedMNIST(images=dataset.data, bc=bc, labels=dataset.targets)
        else:
            dataset = ProcessedMNIST(images=dataset.data[indices], bc=bc, labels=dataset.targets[indices])

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
