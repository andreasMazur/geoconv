import torch

@torch.jit.script
def tensor_scatter_nd_update_(tensor : torch.Tensor, indices : torch.Tensor, updates : torch.Tensor):
    """In-place updates the tensor at the specified indices with the given updates.

    Parameters
    ----------
    tensor: torch.Tensor
        The tensor to update.
    indices: torch.Tensor
        The indices where the updates should be applied.
    updates: torch.Tensor
        The values to update the tensor with.

    Returns
    -------
    torch.Tensor:
        The updated tensor.
    """
    indices = indices.long()

    flat_indices = list(indices.view(-1, tensor.dim()).t())
    flat_updates = updates.view(-1)
    tensor.index_put_(flat_indices, flat_updates, accumulate=False)

@torch.jit.script
def tensor_scatter_nd_add_(tensor : torch.Tensor, indices : torch.Tensor, updates : torch.Tensor):
    """In-place adds `updates` to `tensor` at `indices`.

    Parameters
    ----------
    tensor: torch.Tensor
        The tensor to which the updates should be added.
    indices: torch.Tensor
        The indices where the updates should be added.
    updates: torch.Tensor
        The values to add to the tensor.

    Returns
    -------
    torch.Tensor:
        The updated tensor.
    """
    indices = indices.long()

    flat_indices = list(indices.view(-1, tensor.dim()).t())
    flat_updates = updates.view(-1)
    tensor += torch.zeros_like(tensor).index_put_(flat_indices, flat_updates, accumulate=True)

@torch.jit.script
def histogram_fixed_width_bins(values : torch.Tensor, value_range : torch.Tensor, n_bins : int, dtype : torch.dtype = torch.float32) -> torch.Tensor:
    """Bins the given values for use in a histogram.

    Given the tensor `values`, this function returns a rank 1 `Tensor`
    representing the indices of a histogram into which each element 
    of `values` would be binned. The bins are equal width and 
    determined by the argument `value_range` and `n_bins`.

    Parameters
    ----------
    values: torch.Tensor
        A rank 1 tensor of shape (n_values,) containing the values to bin.
    value_range: torch.Tensor
        Shape [2] `Tensor` of same `dtype` as `values`.
        values <= value_range[0] will be mapped to hist[0],
        values >= value_range[1] will be mapped to hist[-1].
    n_bins: int
        The number of bins to use.
    dtype: torch.dtype
        The data type of the returned histogram.
    
    Returns
    -------
    torch.Tensor:
        A rank 1 tensor of shape (n_values,) containing the indices of the histogram bins.
    """
    shape = values.shape
    values = values.reshape(-1)
    n_bins_torch = torch.tensor(n_bins, dtype=dtype, device=values.device)

    # Map tensor values that fall within value_range to [0, 1]
    scaled_values = torch.true_divide(values - value_range[0], value_range[1] - value_range[0])

    # Map tensor values within the open interval value_range to {0, ..., n_bins - 1}
    # values outside the open interval will be 0 or less, or n_bins or more
    indices = torch.floor(n_bins_torch * scaled_values)

    # Clip edge cases (e.g. value = value_range[1]) or "outliers".
    indices = torch.clamp(indices, min=0, max=n_bins_torch - 1)

    return indices.to(torch.int32).reshape(shape)