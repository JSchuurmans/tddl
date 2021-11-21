import torch

def calculate_error(
    original, 
    approximation,
    **kwargs,
):
    with torch.no_grad():
        return torch.norm(original-approximation, **kwargs)