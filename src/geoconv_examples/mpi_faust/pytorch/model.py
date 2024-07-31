from geoconv.pytorch.layers.angular_max_pooling import AngularMaxPooling
from geoconv.pytorch.layers.conv_dirac import ConvDirac
from geoconv.pytorch.layers.conv_geodesic import ConvGeodesic
from geoconv.pytorch.layers.conv_zero import ConvZero

from torch import nn
from torcheval.metrics.functional import multiclass_accuracy

import torch
import sys


def custom_exp_scheduler(opt, step, decay_rate=0.95, decay_steps=500):
    """Smooth exponential scheduler"""
    opt.param_groups[0]["lr"] = opt.param_groups[0]["initial_lr"] * decay_rate ** (step / decay_steps)


def print_mem():
    mem = torch.cuda.memory_allocated()
    max_mem = torch.cuda.max_memory_allocated()
    return f"{mem / 1024 ** 2:.3f} MB / Max memory: {max_mem / 1024 ** 2:.3f} MB"


class Normalization(nn.Module):

    def __init__(self, dataset):
        super().__init__()
        self.mean, self.var = 0, 0
        n_samples = 0

        for s in dataset:
            n_samples += s.shape[0]
            s = torch.sum(s, dim=-2)
            self.mean += s
        self.mean = self.mean / n_samples

        dataset.reset()
        for s in dataset:
            self.var += torch.sum((s - self.mean) ** 2, dim=-2)
        self.var = self.var / (n_samples - 1)

    def forward(self, inputs):
        return (inputs - self.mean) / self.var


class Imcnn(nn.Module):
    def __init__(self,
                 signal_dim,
                 kernel_size,
                 template_radius,
                 adapt_data,
                 layer_conf=None,
                 variant="dirac",
                 segmentation_classes=-1):
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

        #############
        # ISC Layers
        #############
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
        if segmentation_classes:
            self.output_dense = nn.Linear(in_features=self.output_dims[-1], out_features=segmentation_classes)
        else:
            self.output_dense = nn.Linear(in_features=self.output_dims[-1], out_features=6890)

    def forward(self, inputs):
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

    def train_loop(self,
                   dataset,
                   loss_fn,
                   opt,
                   decay_rate=0.95,
                   decay_steps=500,
                   verbose=True,
                   epoch=None,
                   prev_steps=None,
                   use_lr_decay=False):
        self.train()
        epoch_accuracy = 0.
        epoch_loss = 0.
        mean_accuracy = 0.
        mean_loss = 0.

        for step, ((signal, bc), gt) in enumerate(dataset):
            pred = self([signal, bc])
            loss = loss_fn(pred, gt)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if use_lr_decay:
                custom_exp_scheduler(opt, prev_steps + step, decay_rate=decay_rate, decay_steps=decay_steps)

            # Statistics
            epoch_accuracy = epoch_accuracy + multiclass_accuracy(pred, gt).detach()
            epoch_loss = epoch_loss + loss.detach()

            # I/O
            mean_accuracy = epoch_accuracy / (step + 1)
            mean_loss = epoch_loss / (step + 1)
            if verbose:
                sys.stdout.write(
                    f"\rEpoch: {epoch} - "
                    f"Training step: {step} - "
                    f"Loss {mean_loss:.4f} - "
                    f"Accuracy {mean_accuracy:.4f}"
                    f" - Memory: {print_mem()}"
                )

        return {"epoch_loss": mean_loss, "epoch_accuracy": mean_accuracy}

    def validation_loop(self, dataset, loss_fn, verbose=True):
        self.eval()

        with torch.no_grad():
            val_loss = 0.
            val_accuracy = 0.

            for step, ((signal, bc), gt) in enumerate(dataset):
                pred = self([signal, bc])

                # Statistics
                val_accuracy = val_accuracy + multiclass_accuracy(pred, gt).detach()
                val_loss = val_loss + loss_fn(pred, gt).detach()

            # I/O
            mean_accuracy = val_accuracy / (step + 1)
            mean_loss = val_loss / (step + 1)
            if verbose:
                sys.stdout.write(
                    f" - Val.-Loss: {mean_loss:.4f} - "
                    f"Val.-Accuracy: {mean_accuracy:.4f}"
                )
        return {"val_epoch_loss": mean_loss, "val_epoch_accuracy": mean_accuracy}
