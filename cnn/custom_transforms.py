import numpy as np
import torch


class ToTensor:
    def __init__(self, dtype: torch.dtype, device: torch.device):
        self.dtype = dtype
        self.device = device

    def __call__(self, x: np.ndarray):
        # Normalize between 0 and 1
        # min = np.amin(x)
        max = np.amax(x)
        lower = 0.
        upper = 1.
        x *= upper / max

        return torch.from_numpy(x).type(self.dtype).to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
