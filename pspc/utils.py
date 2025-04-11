
import random
from typing import Any, Callable

import numpy as np
import torch

from pspc.dataset import ImageFolderDataset

class AttributeDict(dict):
    """Dictionary wrapper for attribute access"""

    def __getattr__(self, name: str) -> Any:
        if name not in self:
            raise AttributeError(name)
        return self[name]
    
    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def set_seed(seed):
    """Set all of the random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def to_img(tensor):
    """Convert a [N,C,H,W] image tensor normalized between [-1,1] to a [0,1], [H,W,C] tensor."""
    return torch.clip(tensor[0].permute(1, 2, 0) / 2. + 0.5, 0, 1)


def edm_t_schedule(n_steps: int, t_min: float=2E-3, t_max: float=80, rho: int=7) -> torch.Tensor:
    """Produce a discretized diffusion schedule, from t_max to t_min with n_steps."""
    step_indices = torch.arange(n_steps, dtype=torch.float)
    t_steps = (t_max ** (1 / rho) + step_indices / (n_steps - 1) * (
                t_min ** (1 / rho) - t_max ** (1 / rho))) ** rho
    return t_steps


def imgs_from_dataset(dataset: ImageFolderDataset) -> torch.Tensor:
    """Construct a data tensor from a dataset."""
    imgs = torch.empty([len(dataset),] + dataset.image_shape, dtype=torch.uint8)
    for i, (img, _) in enumerate(dataset):
        imgs[i] = torch.from_numpy(img)
    return imgs


def default_preprocess_fn(imgs: torch.Tensor):
    """Standard preprocessing function to convert uint8 images into range [-1, 1] data"""
    return imgs.float() / 127.5 - 1


def empirical_pmean(
        z: torch.Tensor, t: torch.Tensor, imgs: torch.Tensor, bs: int=100,
        preprocess_fn: Callable =default_preprocess_fn) -> torch.Tensor:
    """Calculate the posterior mean over the set of imgs given z and t in a fancy one-pass way.

    Args:
        z: A [B, ...] tensor of noisy data
        t: A [B,] tensor indicating the diffusion time
        imgs: A [N, ...] tensor of clean data with the same shape as z.
        bs: The batch size used to iterate across imgs.
        preprocess_fn: A function which transforms imgs to match the diffusion preprocessing.
    """
    N = imgs.shape[0]
    B = z.shape[0]
    assert imgs.shape[1:] == z.shape[1:]

    z_flat = z.reshape(B, -1)
    img_flat = imgs.reshape(N, -1)

    m = torch.full([B, 1], - float('inf'), device=z.device)
    normalizer = torch.zeros([B, 1], device=z.device)
    running_mean = torch.zeros_like(z_flat)
    for start in range(0, N, bs):
        end = min(N, start + bs)
        img_batch = preprocess_fn(img_flat[start:end]).to(z.device)
        z_scores = (z_flat.unsqueeze(1) - img_batch).square().sum(dim=-1) / (-2 * t.view(B, 1) ** 2)
        m_new = torch.maximum(z_scores.max(dim=1, keepdim=True)[0], m)
        probs = (z_scores - m_new).exp()
        m_adj = (m - m_new).exp()
        new_normalizer = normalizer * m_adj + probs.sum(dim=1, keepdim=True)
        batch_mean = (img_batch * probs.view(B, -1, 1, )).sum(dim=1)
        running_mean = (running_mean * normalizer.view(B, 1, ) * m_adj.view(B,1) + batch_mean)
        running_mean /= new_normalizer.view(B, 1)
        m = m_new
        normalizer = new_normalizer
    return running_mean.reshape_as(z)


def masked_crop(data: torch.Tensor, mask: torch.Tensor):
    """Crop data using a boolean mask

    Args:
        data: A [N, C, H, W] tensor of image data.
        mask: A [H, W] boolean mask.

    Returns:
        A [N, D] cropped matrix where D is the number of included dimensions.
    """
    return data.flatten(start_dim=2)[:, :, mask.view(-1)].view(data.shape[0], -1)


def masked_uncrop(cropped_data, mask):
    """Revert cropped, flattened data into standard image shape.

    Args:
        cropped_data: A [N, D] tensor of cropped, flattened data
        mask: A [H, W] boolean mask.
    Returns:
        A [N, C, H, W] data whith cropped_data in the positions indicated by mask and zeros elsewhere.
    """
    H, W = mask.shape
    N, D = cropped_data.shape
    n_masked_pixels = mask.int().sum().item()
    assert D % n_masked_pixels == 0
    C = int(D / n_masked_pixels)
    out = torch.zeros((N, C, H*W), device=cropped_data.device)
    out[:, :, mask.view(-1)] = cropped_data.view(N, C, -1)
    return out.view(N, C, H, W)
