
# File containing multiple diffusion denoisers

from abc import abstractmethod
import pickle
from typing import Optional

import torch
import torch.nn as nn

from pspc.dataset import ImageFolderDataset
from pspc.utils import AttributeDict, default_preprocess_fn, imgs_from_dataset, empirical_pmean, masked_crop, masked_uncrop, edm_t_schedule


DENOISER_NAMES = ['empirical', 'network', 'gaussian', 'pspc_sq', 'pspc_flex']

class BaseDenoiser(nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Denoise noisy data z corresponding to diffusion time t

        Args:
            z: A [N, C, H, W] batch of noisy images.
            t: A [N,] batch of diffusion times.

        Returns:
            [N, C, H, W] shape denoised versions of z.
        """

    @classmethod
    def construct_by_class_name(cls, class_name: str, **kwargs):
        if class_name == 'empirical':
            return EmpiricalDenoiser(**kwargs)
        elif class_name == 'network':
            return NetworkDenoiser(**kwargs)
        elif class_name == 'gaussian':
            return GaussianDenoiser(**kwargs)
        elif class_name == 'pspc_sq':
            return PSPCSquareDenoiser(**kwargs)
        elif class_name == 'pspc_flex':
            return PSPCFlexDenoiser(**kwargs)
        else:
            raise ValueError(f'Unrecognized denoiser class_name: {class_name}')



class EmpiricalDenoiser(BaseDenoiser):
    """Optimal Denoiser under the multi-Dirac data distribution assumption."""

    def __init__(self, dataset: ImageFolderDataset, empirical_bs=100):
        """Construct the optimal denoiser for the provided dataset.

        Args:
            dataset: The data distribution which you want the optimal denoiser for.
        """
        super().__init__()
        self.imgs = imgs_from_dataset(dataset)
        self.empirical_bs = empirical_bs

    def forward(self, z, t):
        return empirical_pmean(z, t, self.imgs, bs=self.empirical_bs)


class NetworkDenoiser(BaseDenoiser):
    """Neural Network based Denoiser."""

    def __init__(self, network_ckpt: str):
        """Initialize NetworkDenoiser from the network checkpoint.

        Args:
            network_ckpt: Path to a network checkpoint dictionary pickle. Assume network checkpoint has the model stored
                under the 'ema' key. Network should have the same interface as these denoisers.
        """
        super().__init__()
        with open(network_ckpt, 'rb') as f:
            self.net = pickle.load(f)['ema']

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(z, t.view(z.shape[0], 1, 1, 1))


class GaussianDenoiser(BaseDenoiser):
    """Implementation of the optimal denoiser under a Gaussian data assumption."""

    def __init__(self, dataset: ImageFolderDataset, rank: Optional[int] = None):
        """Construct the optimal Gaussian denoiser for the provided dataset

        Args:
            dataset: The data distribution which you want the optimal gaussian denoiser for.
            rank: The number of dimensions to retain from the SVD decomposition.
        """
        super().__init__()
        imgs = default_preprocess_fn(imgs_from_dataset(dataset).float().view(len(dataset), -1))
        self.D = imgs.shape[-1] if rank is None else rank
        assert self.D <= imgs.shape[-1]
        self.register_buffer('mean', imgs.mean(dim=0, keepdim=True))
        cov_svd = torch.linalg.svd(torch.cov(imgs.T, correction=0), full_matrices=False)
        self.register_buffer('U', cov_svd[0][None, :, :self.D])
        self.register_buffer('S', cov_svd[1][None, :self.D])
        self.register_buffer('Vh', cov_svd[2][None, :self.D, :])

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        N = z.shape[0]
        s_aug = self.S / (self.S.unsqueeze(0) + t.view(N, 1) ** 2)
        augmented_cov = self.U @ torch.diag_embed(s_aug) @ self.Vh
        diff = (z.view(N, -1) - self.mean).unsqueeze(-1)
        return (self.mean + (augmented_cov @ diff).squeeze(-1)).view_as(z)


class PSPCDenoiser(BaseDenoiser):

    def __init__(self, dataset: ImageFolderDataset):
        super().__init__()
        self.imgs = imgs_from_dataset(dataset)
        self.C = dataset.num_channels
        self.res = dataset.resolution

    @abstractmethod
    def get_mask(self, x: int, y: int, t: float) -> torch.Tensor:
        """Return the correct cropping mask for spatial position (x,y) and diffusion time t.

        Args:
            x: The x coordinate for which to get the mask. Between 0 and self.res.
            y: The y coordinate for which to get the mask. Between 0 and self.res.
            t: The diffusion time. Note, this PSPC implementation doesn't work with varying diffusion times.

        Returns:
            A [C, H, W] boolean tensor indicating which pixels to include in the patch set.
        """

    def forward(self, z, t):
        composite = torch.zeros_like(z)
        normalizer = torch.zeros_like(z)
        t_val = t[0].item()

        for x in range(self.res):
            for y in range(self.res):
                # Get the cropping matrix C
                mask = self.get_mask(x, y, t_val)

                # Skip cases where we don't have a reasonable mask.
                if not mask.any():
                    continue

                # Compute z_C and the patch set.
                cropped_z = masked_crop(z, mask)  # [N, D]
                patch_set = masked_crop(self.imgs, mask)  # [len(dataset), D]

                # Compute the posterior mean over the patch set.
                patch_posterior_mean = empirical_pmean(cropped_z, t, patch_set)

                # Add the posterior mean to the composite.
                composite += masked_uncrop(patch_posterior_mean, mask)
                # Step the normalizer inside the cropped area.
                normalizer += masked_uncrop(torch.ones_like(patch_posterior_mean), mask)

        # Normalize the composite and return.
        return composite / normalizer


CIFAR_PATCH_SCHEDULE = '32,32,32,32,32,32,32,23,15,11,7,5,3,3,3,3,3,3'

class PSPCSquareDenoiser(PSPCDenoiser):
    """PSPC Denoiser using square patches."""

    def __init__(self, dataset: ImageFolderDataset, patch_sizes: torch.Tensor, diffusion_kwargs=None):
        """Construct a PSPC Square Denoiser.

        Args:
            dataset: The empirical dataset to compute patch set posterior means with.
            patch_sizes: A [L,] tensor of patch sizes, corresponding to L values of t.
            diffusion_kwargs: arguments to specify the diffusion time points.
        """
        super().__init__(dataset)
        self.patch_sizes = patch_sizes
        diffusion_kwargs = AttributeDict() if diffusion_kwargs is None else diffusion_kwargs
        diffusion_kwargs.n_steps = patch_sizes.shape[0]
        self.t_thresholds = edm_t_schedule(**diffusion_kwargs)

    def patch_size_schedule(self, t: float) -> int:
        """Determine the appropriate patch_size for a given diffusion time."""
        nearest = (self.t_thresholds - t).abs().argmin()
        return self.patch_sizes[nearest]

    def get_mask(self, x: int, y: int, t: float) -> torch.Tensor:
        """Get a square cropping matrix with upper left corner at point x,y."""
        patch_size = self.patch_size_schedule(t)
        mask = torch.zeros((self.res, self.res), dtype=torch.bool)

        # Return an empty mask if we don't have room for a full patch.
        if x + patch_size > self.res or y+patch_size > self.res:
            return mask

        mask[y:y+patch_size, x:x+patch_size] = True
        return mask


def mask_from_heatmap(heatmap: torch.Tensor, threshold: float) -> torch.Tensor:
    """Construct a cropping mask from a heatmap and a cumulative threshold.

    Args:
        heatmap: A [H, W] tensor capturing some notion of sensitivity.
        threshold: The cumulative sensitivity which should be captured in the mask.

    Returns:
          A [H,W] boolean mask tensor which contains threshold percent of the total of heatmap.
    """
    inv_threshold = 1. - threshold
    normalized = heatmap / heatmap.sum()
    normalized = normalized.view(-1)
    sorted_vals, indices = normalized.sort(descending=False)
    sorted_vals = torch.cumsum(sorted_vals, dim=0)
    mask = sorted_vals >= inv_threshold
    assert mask.any()
    mask[indices] = mask.clone()
    return mask.reshape_as(heatmap)


CIFAR_THRESHOLD_SCHEDULE = '1.,1.,1.,1.,1.,1.,1.,0.7,0.7,0.5,0.4,0.4,0.4,0.4,0.3,0.3,0.3,0.3'

class PSPCFlexDenoiser(PSPCDenoiser):
    """PSPC Denoiser using flexible patches determined by gradient heatmaps."""

    def __init__(
            self, dataset: ImageFolderDataset, heatmaps: torch.Tensor, grad_thresholds: torch.Tensor,
            diffusion_kwargs=None
    ):
        """Initialize the PSPC-Flex Denoiser

        Args:
            dataset: The empirical dataset to compute patch set posterior means on.
            heatmaps: A tensor of shape [L, res, res, res, res] where heatmap[t_idx, y, x] corresponds to the average
                gradient heatmap for diffusion time t_idx and pixel (x,y).
            grad_thresholds: A shape [L] tensor where each entry is between 0. and 1. indicating the percentage of the
                gradient heatmap for that diffusion time which should be used to generate the cropping mask.
            diffusion_kwargs: arguments to specify the diffusion time points.
        """
        super().__init__(dataset)
        self.heatmaps = heatmaps
        self.grad_thresholds = grad_thresholds
        diffusion_kwargs = AttributeDict() if diffusion_kwargs is None else diffusion_kwargs
        diffusion_kwargs.n_steps = grad_thresholds.shape[0]
        self.t_thresholds = edm_t_schedule(**diffusion_kwargs)

    def closest_t_idx(self, t: float) -> int:
        """Determine the appropriate patch_size for a given diffusion time."""
        return (self.t_thresholds - t).abs().argmin()

    def get_mask(self, x: int, y: int, t: float) -> torch.Tensor:
        t_idx = self.closest_t_idx(t)
        threshold, heatmap = self.grad_thresholds[t_idx], self.heatmaps[t_idx]

        # If the threshold is 1. we only have to run the computation at 0,0 since all pixels will be included.
        if threshold == 1.0 and (x > 0 or y > 0):
            return torch.zeros((self.res, self.res), dtype=torch.bool)
        return mask_from_heatmap(heatmap[y, x], threshold)
    
    """The code above this point is provided for implementation context and to provide background material
    for the auto code generation (namely OpenAI's codex) to modify the code below this line."""

    class AutoDenoiser(BaseDenoiser):
        """Candidate denoiser under the auto generation of denoisers beam search."""

    def __init__(self, dataset: ImageFolderDataset, empirical_bs=100):
        """Construct the denoiser for the provided dataset.

        Args:
            dataset: The data distribution which you want the denoiser for.
        """
        super().__init__()
        self.imgs = imgs_from_dataset(dataset)
        self.empirical_bs = empirical_bs

    """The function signature here may also be adjusted, but remember to also adjust denoise and utils/sample accordingly."""
    def forward(self, z, t):
        """Denoise the noisy data z corresponding to diffusion time t.  This is a reference implementation of 
        the denoiser which is to be rewritten and improved by the beam search."""
        return empirical_pmean(z, t, self.imgs, bs=self.empirical_bs)