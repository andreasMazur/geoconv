from geoconv_examples.mpi_faust.pytorch.model import Imcnn

from torch import nn


class SegImcnn(nn.Module):
    def __init__(self, adapt_data):
        super().__init__()

        self.model = Imcnn(
            signal_dim=3,  # Use 3D-coordinates as input
            kernel_size=(5, 8),
            adapt_data=adapt_data,
            layer_conf=[(96, 1)],
            variant="dirac",
            segmentation=10,
            template_radius=0.3274390161214167  # TODO: Read from dataset
        )

    def forward(self, x):
        return self.model.forward(x)

    def train_loop(self, dataset, loss_fn, optimizer, verbose=True, epoch=None):
        self.model.train_loop(dataset, loss_fn, optimizer, verbose=verbose, epoch=epoch)
