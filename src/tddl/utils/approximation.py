import torch

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
        return torch.norm(original-approximation, **kwargs)


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
        return torch.norm(original-approximation, **kwargs) / torch.numel(original)


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
        return torch.norm(original-approximation, **kwargs) / torch.norm(original, **kwargs)