import torch

def scipy_softmax(x):
    """Softmax like in scipy"""
    e_x = torch.exp(x)
    return e_x / torch.sum(e_x)