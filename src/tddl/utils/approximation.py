import torch
import tensorly as tl
tl.set_backend('pytorch')


def calculate_error(
    original, 
    approximation,
    **kwargs,
):
    """
    Returns:
        The norm of the difference with tracking gradients.
    """
    with torch.no_grad():
        return torch.norm(
            original-approximation.to(device=original.device), 
            **kwargs
        )


def calculate_scaled_error(
    original, 
    approximation,
    **kwargs,
):
    """
    Returns:
        The error and scales with the number of elements in the original tensor.
    """
    with torch.no_grad():
        return torch.norm(
            original-approximation.to(device=original.device), 
            **kwargs) / torch.numel(original)


def calculate_relative_error(
    original,
    approximation,
    **kwargs,
):
    """
    Returns
        The error relative to the norm of the original tensor.
    """
    with torch.no_grad():
        return torch.norm(original-approximation.to(device=original.device), 
        **kwargs) / torch.norm(original, **kwargs)


def relative_error(pre_weight, dec_weight):
    with torch.no_grad():
        return tl.norm(pre_weight-dec_weight)/tl.norm(pre_weight)
