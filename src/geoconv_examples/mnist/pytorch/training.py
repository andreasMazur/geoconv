import torch.utils
from geoconv.pytorch.layers import ConvGeodesic
from geoconv.pytorch.layers import ConvZero
from geoconv.pytorch.layers import ConvDirac
from geoconv.pytorch.layers import AngularMaxPooling
from geoconv.utils.data_generator import read_template_configurations
from geoconv.utils.prepare_logs import process_logs
from geoconv_examples.mnist.pytorch.dataset import load_preprocessed_mnist

import torch
from torch.utils.tensorboard import SummaryWriter
import os
import os.path as osp
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

from tqdm import tqdm


class MNISTClassifier(torch.jit.ScriptModule):
    def __init__(
        self,
        template_radius,
        input_shapes,
        variant=None,
        isc_layer_dims=None,
        hidden_layer_dims=None,
        hidden_activation=torch.nn.LeakyReLU,
    ):
        super().__init__()

        if isc_layer_dims is None:
            isc_layer_dims = [128]

        if variant is None or variant == "dirac":
            self.layer_type = ConvDirac
        elif variant == "geodesic":
            self.layer_type = ConvGeodesic
        elif variant == "zero":
            self.layer_type = ConvZero
        else:
            raise RuntimeError(
                "Select a layer type from: ['dirac', 'geodesic', 'zero']"
            )

        in_shapes = [input_shapes]
        for n in isc_layer_dims:
            tmp_shape = ((*(input_shapes[0][:-1]), n), input_shapes[1])
            in_shapes.append(tmp_shape)

        self.convs = torch.nn.ModuleList(
            [
                self.layer_type(
                    amt_templates=n,
                    template_radius=template_radius,
                    activation="leaky_relu",
                    rotation_delta=1,
                    input_shape=shape,
                )
                for n, shape in zip(isc_layer_dims, in_shapes)
            ]
        )
        self.amp = AngularMaxPooling()
        self.hidden_layers = torch.nn.ModuleList([])

        in_dim = isc_layer_dims[-1]
        for i in hidden_layer_dims:
            self.hidden_layers.append(torch.nn.Linear(in_dim, i))
            self.hidden_layers.append(hidden_activation())
            in_dim = i

        self.output_layer = torch.nn.Linear(in_dim, 10)

    def forward(self, inputs, **kwargs):
        signal, bc = inputs
        for layer in self.convs:
            signal = layer(
                signal, bc
            )  # [bathch_size, n_verts, n_rotations, n_templates]
            signal = self.amp(signal)  # [batch_size, n_verts, n_templates]
        signal = torch.max(signal, dim=-2).values  # [batch_size, n_templates, 10]
        for layer in self.hidden_layers:
            signal = layer(signal)
        return self.output_layer(signal)  # [batch_size, 10]


def training(
    bc_path,
    logging_dir,
    k=5,
    template_configurations=None,
    variant=None,
    batch_size=8,
    isc_layer_dims=None,
    hidden_layer_dims=None,
    hidden_activation=torch.nn.LeakyReLU,
):

    if not osp.isfile(bc_path):
        from geoconv_examples.mnist.preprocess import preprocess

        procs = max(os.cpu_count() - 2, 1)
        preprocess(bc_path, procs)

    # Create logging dir
    os.makedirs(logging_dir, exist_ok=True)

    # Setup default layer parameterization if not given
    if isc_layer_dims is None:
        isc_layer_dims = [128]

    # Prepare k-fold cross-validation
    if template_configurations is None:
        template_configurations = read_template_configurations(bc_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for n_radial, n_angular, template_radius in template_configurations:
        csv_file_names = []

        full_dataset = load_preprocessed_mnist(
            bc_path, n_radial, n_angular, template_radius, "all"
        )
        full_indices = np.arange(len(full_dataset))

        kfold = KFold(n_splits=k, shuffle=True)

        for k_idx, (train_idxs, val_idxs) in enumerate(kfold.split(full_indices)):
            # Get k-th fold
            train_dataset = torch.utils.data.DataLoader(
                full_dataset.dataset.subset(train_idxs),
                batch_size=batch_size,
                shuffle=True,
            )
            val_dataset = torch.utils.data.DataLoader(
                full_dataset.dataset.subset(val_idxs),
                batch_size=batch_size,
                shuffle=True,
            )

            signal_shape, bary_shape = full_dataset.dataset.get_shapes()
            shapes = ([batch_size, *signal_shape], [batch_size, *bary_shape])

            imcnn = MNISTClassifier(
                template_radius,
                variant=variant,
                isc_layer_dims=isc_layer_dims,
                input_shapes=shapes,
                hidden_layer_dims=hidden_layer_dims,
                hidden_activation=hidden_activation,
            )
            imcnn.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(imcnn.parameters(), lr=1e-3, foreach=True)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

            exp_number = f"{k_idx}__{n_radial}_{n_angular}_{template_radius}"
            csv_file_name = f"{logging_dir}/training_{exp_number}.log"
            csv_file_names.append(csv_file_name)

            writer = SummaryWriter(log_dir=f"{logging_dir}/training_{exp_number}")
            best_val_los = float("inf")
            save_path = f"{logging_dir}/saved_imcnn_{exp_number}.pth"

            # Training loop
            epochs = 10
            train_log = []
            for epoch in range(epochs):
                imcnn.train()
                train_loss, correct, total = 0, 0, 0

                train_loader = tqdm(
                    train_dataset,
                    desc=f"Epoch {epoch+1}/{epochs}",
                    unit="batch",
                    leave=True,
                )
                for (images, bc), labels in train_loader:
                    images, bc, labels = (
                        images.to(device),
                        bc.to(device),
                        labels.to(device),
                    )
                    optimizer.zero_grad()
                    output = imcnn((images, bc))
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * images.size(0)
                    _, predicted = output.max(1)
                    correct += (predicted.eq(labels)).sum().item()
                    total += labels.size(0)

                    avg_loss = train_loss / total
                    acc = correct / total
                    train_loader.set_postfix(loss=avg_loss, acc=acc)

                train_loss /= total
                train_acc = correct / total

                lr_scheduler.step()

                # Validation loop
                imcnn.eval()
                val_loss, correct, total = 0, 0, 0
                with torch.no_grad():
                    val_loader = tqdm(
                        val_dataset, desc=f"Validation", unit="batch", leave=True
                    )
                    for (images, bc), labels in val_loader:
                        images, bc, labels = (
                            images.to(device),
                            bc.to(device),
                            labels.to(device),
                        )
                        output = imcnn((images, bc))
                        loss = criterion(output, labels)

                        val_loss += loss.item() * images.size(0)
                        _, predicted = output.max(1)
                        correct += (predicted.eq(labels)).sum().item()
                        total += labels.size(0)

                        avg_loss = val_loss / total
                        acc = correct / total
                        val_loader.set_postfix(val_loss=avg_loss, val_acc=acc)

                val_loss /= total
                val_acc = correct / total

                # Logging
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Accuracy/train", train_acc, epoch)
                writer.add_scalar("Accuracy/val", val_acc, epoch)

                train_log.append([epoch, train_loss, train_acc, val_loss, val_acc])

                if val_loss < best_val_los:
                    best_val_los = val_loss
                    torch.save(imcnn.state_dict(), save_path)

        # Save training logs
        df = pd.DataFrame(
            train_log,
            columns=["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"],
        )
        df.to_csv(csv_file_name, index=False)
        writer.close()


print("###### MNIST training ###")
print("## Geodesic")
training(
    "/tmp/mnist",
    "/tmp/geolog/geodesic",
    variant="geodesic",
    batch_size=64,
    isc_layer_dims=[32, 64, 128],
    hidden_layer_dims=[128, 64, 32],
)
