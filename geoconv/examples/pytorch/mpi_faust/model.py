from geoconv.layers.pytorch.angular_max_pooling import AngularMaxPooling
from geoconv.layers.pytorch.conv_geodesic import ConvGeodesic
from geoconv.layers.pytorch.conv_zero import ConvZero
from geoconv.layers.pytorch.conv_dirac import ConvDirac
from torch import nn

import torch


class Normalization(nn.Module):

    def __init__(self, dataset):
        super().__init__()
        self.mean, self.var = 0, 0
        n_samples = 0
        for s in dataset:
            s = torch.sum(s, dim=-2)
            self.mean += s
            self.var += (s - self.mean) ** 2
            n_samples += 1
        self.mean = self.mean / n_samples
        self.var = self.var / n_samples

    def forward(self, inputs):
        return (inputs - self.mean) / self.var


class Imcnn(nn.Module):
    def __init__(self, signal_dim, kernel_size, template_radius, adapt_data, layer_conf=None, variant="dirac"):
        super().__init__()
        self.signal_dim = signal_dim
        self.kernel_size = kernel_size
        self.template_radius = template_radius

        if variant == "dirac":
            self.layer_type = ConvDirac
        elif variant == "geodesic":
            self.layer_type = ConvGeodesic
        elif variant == "zero":
            self.layer_type = ConvZero
        else:
            raise RuntimeError("Select a layer type from: ['dirac', 'geodesic', 'zero']")

        if layer_conf is None:
            self.output_dims = [96, 256, 384, 384]
            self.rotation_deltas = [1 for _ in range(len(self.output_dims))]
        else:
            self.output_dims, self.rotation_deltas = list(zip(*layer_conf))
        self.downsize_dim = 64
        self.input_dims = [self.downsize_dim]
        self.input_dims.extend(self.output_dims)

        #################
        # Handling Input
        #################
        self.normalize = Normalization(adapt_data)
        self.downsize_dense = nn.Linear(in_features=signal_dim, out_features=self.downsize_dim)
        self.downsize_activation = nn.ReLU()
        self.downsize_bn = nn.BatchNorm1d(num_features=self.downsize_dim)

        ##################
        # Global Features
        ##################
        self.do_layers = nn.ModuleList()
        self.isc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.amp_layers = nn.ModuleList()
        for idx in range(len(self.output_dims)):
            self.do_layers.append(nn.Dropout(p=0.2))
            self.isc_layers.append(
                self.layer_type(
                    input_shape=[(None, self.input_dims[idx]), (None, kernel_size[0], kernel_size[1], 3, 2)],
                    amt_templates=self.output_dims[idx],
                    template_radius=self.template_radius,
                    activation="relu",
                    rotation_delta=self.rotation_deltas[idx]
                )
            )
            self.bn_layers.append(nn.BatchNorm1d(num_features=self.output_dims[idx]))
            self.amp_layers.append(AngularMaxPooling())

        #########
        # Output
        #########
        self.output_dense = nn.Linear(in_features=self.output_dims[-1], out_features=6890)

    def forward(self, inputs, orientations=None):
        #################
        # Handling Input
        #################
        signal, bc = inputs
        signal = self.normalize(signal)
        signal = self.downsize_dense(signal)
        signal = self.downsize_activation(signal)
        signal = self.downsize_bn(signal)
        ###############
        # Forward pass
        ###############
        for idx in range(len(self.output_dims)):
            signal = self.do_layers[idx](signal)
            signal = self.isc_layers[idx]([signal, bc])
            signal = self.amp_layers[idx](signal)
            signal = self.bn_layers[idx](signal)

        #########
        # Output
        #########
        return self.output_dense(signal)


def print_mem(note=""):
    mem = torch.cuda.memory_allocated()
    print(f"{note}: {mem / 1024 ** 2} MB / Max memory: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")
