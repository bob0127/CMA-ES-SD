import torch
import numpy as np

def flatten_latent(latent_tensor: torch.Tensor) -> np.ndarray:
    """
    將 latent tensor (e.g. [1, 4, 64, 64]) 展平成 1D numpy array。
    """
    return latent_tensor.view(-1).cpu().numpy()

def reshape_latent(flat_array: np.ndarray, shape=(1, 4, 64, 64)) -> torch.Tensor:
    """
    將 1D numpy array 還原成 latent tensor。
    """
    return torch.tensor(flat_array, dtype=torch.float32).view(*shape)