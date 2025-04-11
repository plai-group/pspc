
# File which runs denoisers on forward or reverse-process z.

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

@click.command()
# Generic options.
@click.option('--outdir', help='Where to save the results', metavar='DIR', type=str, required=True)
@click.option('--t_idx', help='Diffusion time index to denoise', metavar=int, type=click.IntRange(min=0), required=True)
@click.option('--denoiser', help='Which denoiser class to use', metavar='STR', type=click.Choice(DENOISER_NAMES), default='pspc_sq', show_default=True)
@click.option('--n_samples', help='How many noisy z to denoise.', metavar='INT', type=int, default=50)
@click.option('--batch', help='Batch size for sampling', metavar='INT', type=int, default=50)
@click.option('--z', help='Optional set of precomputed z. Shape [n_steps, N, C, H, W]', metavar='NPY', type=str)
@click.option('--dim', help='Image size', metavar='INT', type=int, default=32)
@click.option('--device', help='Processing device', metavar='cpu|cuda', type=click.Choice(['cpu', 'cuda']), default='cpu')
@click.option('--seed', help='Random seed for sampling', metavar='INT', type=int, default=0)
# Diffusion schedule options
@click.option('--n_steps', help='How many diffusion timesteps to include in the sampling process', metavar='INT', default=18)
@click.option('--t_max', help='Maximum diffusion time', metavar='FLOAT', type=float, default=80.)
@click.option('--t_min', help='Minimum diffusion time', metavar='FLOAT', type=float, default=2E-3)
@click.option('--rho', help='Exponent for diffusion schedule', metavar='INT', type=int, default=7)
# Dataset options:
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
    assert opts.t_idx < opts.n_steps
    t = edm_t_schedule(**diffusion_kwargs)[opts.t_idx]

    dataset = None
    if opts.data is not None:
        dataset = ImageFolderDataset(opts.data, xflip=opts.xflip)

    # Get z.
    if opts.z is not None:
        z = torch.from_numpy(np.load(opts.z))[opts.t_idx]
        assert z.shape[0] >= opts.n_samples
        assert z.shape[-1] == opts.dim
        z = z[:opts.n_samples]
    elif dataset is not None:
        z = torch.empty(opts.n_samples, 3, opts.dim, opts.dim)
        for i in range(opts.n_samples):
            data_idx = np.random.randint(len(dataset))
            x = torch.tensor(dataset[data_idx][0]) / 127.5 - 1
            z[i] = x + torch.randn_like(x) * t
    else:
        raise Exception('Either --z or --data must be specified.')

    # Build the denoiser.
    denoiser_kwargs = AttributeDict()
    if opts.denoiser == 'network':
        denoiser_kwargs.update(network_ckpt=opts.network_pkl)
    else:
        assert dataset is not None
        denoiser_kwargs.update(dataset=dataset)

    if opts.denoiser == 'gaussian':
        denoiser_kwargs.update(rank=opts.rank)
    elif opts.denoiser == 'pspc_sq':
        denoiser_kwargs.update(patch_sizes=torch.tensor(opts.patch_sizes, dtype=torch.int),
                               diffusion_kwargs=diffusion_kwargs)
    elif opts.denoiser == 'pspc_flex':
        heatmaps = torch.from_numpy(np.load(opts.heatmaps))
        denoiser_kwargs.update(heatmaps=heatmaps, grad_thresholds=torch.tensor(opts.thresholds),
                               diffusion_kwargs=diffusion_kwargs)
    denoiser = BaseDenoiser.construct_by_class_name(opts.denoiser, **denoiser_kwargs).to(opts.device)

    D = torch.empty_like(z)
    for start in tqdm.trange(0, opts.n_samples, opts.batch):
        end = min(start + opts.batch, opts.n_samples)
        batch_size = end - start
        D[start:end] = denoiser(z[start:end].to(opts.device), t.expand(batch_size).to(opts.device)).cpu()

    # Save the samples.
    out_file = os.path.join(opts.outdir, f'{opts.denoiser}_outputs_t_{opts.t_idx:02d}_of_{opts.n_steps:02d}.npy')
    np.save(out_file, D.numpy())


if __name__ == '__main__':
    main()