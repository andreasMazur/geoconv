import torch

class NormalizePointCloud(torch.nn.Module):
    def forward(self, inputs):
        # Move point-cloud into origin
        inputs = inputs - torch.mean(inputs, dim=1, keepdim=True)

        # Scale minimal axis-variance of point-cloud to one. Scale other axes accordingly.
        return inputs / torch.max(torch.std(inputs, dim=1, unbiased=False))