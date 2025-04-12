# Draw samples using one of the denoisers.

import click
from functools import partial
import os
from typing import Callable

import numpy as np
import torch
import tqdm

from pspc.dataset import ImageFolderDataset
from pspc.denoisers import BaseDenoiser, CIFAR_PATCH_SCHEDULE, CIFAR_THRESHOLD_SCHEDULE, DENOISER_NAMES
from pspc.utils import AttributeDict, set_seed, edm_t_schedule

def parse_num_list(s, cls):
    if isinstance(s, list):
        return s
    else:
        return [cls(num) for num in s.split(',')]

def parse_int_list(s):
    return parse_num_list(s, int)

def parse_float_list(s):
    return parse_num_list(s, float)

def sampler(z_prior: torch.Tensor, denoiser: BaseDenoiser, t_steps: torch.Tensor, huen: bool=False) -> torch.Tensor:
    """Draw PF-ODE samples.
    
    Args:
        z_prior: Samples from the diffusion prior. Expected shape [N, C, H, W]
        denoiser: A denoising function which estimates the posterior mean given z and t.
        t_steps: Discretized diffusion timesteps, from high to low.
        huen: Whether to do second-order corrections while sampling. Doubles compute.
    Returns:
        A [N, C, H, W] tensor of samples.
    """
    z = z_prior
    n_steps = t_steps.shape[0]
    N = z_prior.shape[0]
    for i in tqdm.trange(n_steps):
        t = t_steps[i].expand(N, 1, 1, 1)
        t_next = t_steps[i+1].expand(N, 1, 1, 1) if i+1 < n_steps else torch.zeros_like(t)

        D = denoiser(z, t.flatten())
        dz = (z - D) / t
        
        if huen and i+1 < n_steps:
            z_next = z + (t_next - t) * dz
            D = denoiser(z_next, t_next.flatten())
            dz_prime = (z_next - D) / t_next
            dz = 0.5 * (dz + dz_prime)
        
        z = z + (t_next - t) * dz
    return z    

def get_sampling_fnc(sampler_name: str) -> Callable:
    if sampler_name == 'euler':
        return partial(sampler, huen=False)
    elif sampler_name == 'huen':
        return partial(sampler, huen=True)
    else:
        raise NotImplementedError(f'Sampler {sampler_name} unrecognized.')

@click.command()

# Generic options.
@click.option('--outdir', help='Where to save the results', metavar='DIR', type=str, required=True)
@click.option('--denoiser', help='Which denoiser class to use', metavar='STR', type=click.Choice(DENOISER_NAMES), default='pspc_sq', show_default=True)
@click.option('--n_samples', help='How many samples to draw', metavar='INT', type=int, default=50)
@click.option('--batch', help='Batch size for sampling', metavar='INT', type=int, default=50)
@click.option('--sampler', help='PF-ODE solver', metavar='euler|huen', type=click.Choice(['euler', 'huen']), default='euler')
@click.option('--z_init',  help='Initial conditions for sampling', metavar='NPY', type=str)
@click.option('--dim', help='Image size', metavar='INT', type=int, default=32)
@click.option('--device', help='Processing device', metavar='cpu|cuda', type=click.Choice(['cpu', 'cuda']), default='cpu')
@click.option('--seed', help='Random seed for sampling', metavar='INT', type=int, default=0)

# Diffusion schedule options
@click.option('--n_steps', help='How many diffusion timesteps to include in the sampling process', metavar='INT', default=18)
@click.option('--t_max', help='Maximum diffusion time', metavar='FLOAT', type=float, default=80.)
@click.option('--t_min', help='Minimum diffusion time', metavar='FLOAT', type=float, default=2E-3)
@click.option('--rho', help='Exponent for diffusion schedule', metavar='INT', type=int, default=7)

# Empirical Denoiser options:
@click.option('--data', help='Empirical dataset to use for denoising', metavar='ZIP|DIR')
@click.option('--xflip', help='Whether to include xflipped images in the data distribution', metavar='BOOL', type=bool, default=True)

# Network Denoiser options:
@click.option('--network_pkl', help='Path to network checkpoint', metavar='PKL', type=str)

# Gaussian Denoiser options:
@click.option('--rank', help='How many SVD dimensions to use', metavar='INT', type=int, default=None)

# PSPC-Square Denoiser options:
@click.option('--patch_sizes', help='Patch size schedule, in desceding order of t', metavar='LIST', type=parse_int_list, default=CIFAR_PATCH_SCHEDULE)

# PSPC-Flex Denoiser options:
@click.option('--heatmaps', help='Path to gradient sensitivity heatmaps', metavar='NPY', default=str)
@click.option('--thresholds', help='Threshold schedule, in descending order of t', metavar='LIST', type=parse_float_list, default=CIFAR_THRESHOLD_SCHEDULE)

def main(**kwargs):
    opts = AttributeDict(kwargs)

    # Setup.
    set_seed(opts.seed)
    if not os.path.exists(opts.outdir):
        os.makedirs(opts.outdir)

    diffusion_kwargs = AttributeDict(n_steps=opts.n_steps, t_max=opts.t_max, t_min=opts.t_min, rho=opts.rho)

    # Build the denoiser.
    denoiser_kwargs = AttributeDict()
    if opts.denoiser == 'network':
        denoiser_kwargs.update(network_ckpt=opts.network_pkl)
    else:
        dataset = ImageFolderDataset(opts.data, xflip=opts.xflip)
        denoiser_kwargs.update(dataset=dataset)

    if opts.denoiser == 'gaussian':
        denoiser_kwargs.update(rank=opts.rank)
    elif opts.denoiser == 'pspc_sq':
        denoiser_kwargs.update(patch_sizes=torch.tensor(opts.patch_sizes, dtype=torch.int), diffusion_kwargs=diffusion_kwargs)
    elif opts.denoiser == 'pspc_flex':
        heatmaps = torch.from_numpy(np.load(opts.heatmaps))
        denoiser_kwargs.update(heatmaps=heatmaps, grad_thresholds=torch.tensor(opts.thresholds), diffusion_kwargs=diffusion_kwargs)
    denoiser = BaseDenoiser.construct_by_class_name(opts.denoiser, **denoiser_kwargs)
    
    # Construct the sampling schedule and initial z.
    t_steps = edm_t_schedule(**diffusion_kwargs)
    if opts.z_init is not None:
        z = torch.from_numpy(np.load(opts.z_init))
        assert z.shape[0] >= opts.n_samples
        assert z.shape[-1] == opts.dim
        z = z[:opts.n_samples]
    else:
        z = torch.randn(opts.n_samples, 3, opts.dim, opts.dim) * t_steps[0]

    # Pick a sampler, draw some samples.
    sampling_fnc = get_sampling_fnc(opts.sampler)
    samples = torch.empty_like(z)
    for start in tqdm.trange(0, opts.n_samples, opts.batch):
        end = min(start+opts.batch, opts.n_samples)
        samples[start:end] = sampling_fnc(
            z[start:end].to(opts.device), denoiser.to(opts.device), t_steps.to(opts.device))

    # Save the samples.
    out_file = os.path.join(opts.outdir, f'{opts.denoiser}_samples.npy')
    np.save(out_file, samples.cpu().numpy())

if __name__ == '__main__':
    main()