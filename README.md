## Patch Set Posterior Composites

This repository implements a number of diffusion denoiser methods, including the official implementation of **P**atch **S**et **P**osterior **C**omposites (PSPC) which is described in our work

**Towards a Mechanistic Explanation of Diffusion Generalization** <br>
Matthew Niedoba, Berend Zwartsenberg, Kevin Murphy, Frank Wood
<br> https://arxiv.org/abs/2411.19339

## Getting Started

This repo was tested with python `3.9` and pytorch `2.1.0`. The easiest way to get started is to create a  [conda](https://docs.conda.io/en/latest/miniconda.html) environment:

```
conda create -f environment.yml
conda activate pspc
```

## Dataset Setup

We make use of the dataset preprocessing of [EDM](https://github.com/NVlabs/edm/). Please see their repository for data preprocessing instructions.

Additionally, for `pspc_flex`, you need to download the gradient sensitivity heatmaps, available [[here](https://drive.google.com/file/d/1UaiVqSpUauR_HqKieZKnfIGqCxHF_YBd/view?usp=sharing)]

## Denoising

`denoise.py` runs the denoiser of your choice and saves the file to an output `.npy` file. For example

```
python denoise.py --outdir=./tmp --data=PATH_TO_DATASET.zip --denoiser=pspc_sq --t_idx=10 --n_steps=18
```

Will denoise 10 $\mathbf{z}$ drawn from the forward process using PSPC-Square and save them at `./tmp/pspc_sq_outputs_10_of_18.npy`



Instead of using the forward process, you may provide `--z` to specify the noisy latents which you wish to denoise. For example,

```
python denoise.py --outdir=./tmp --data=PATH_TO_DATASET.zip --z=path_to_z.npy --denoiser=pspc_sq --t_idx=10 --n_steps=18
```

Will denoise 10 items from `path_to_z.npy`. We have provided 10,000 latents drawn from the CIFAR-10 forward process and DDPM++ reverse process here:

- CIFAR-10 forward process $\mathbf{z}$ [[Link](https://drive.google.com/file/d/1pSSAfbpBjYNeDkvO-4Euwwlmtx0lDJR4)]
- CIFAR-10 DDPM++ reverse process $\mathbf{z}$ [[Link](https://drive.google.com/file/d/1gyIFPVAogS5yhw9Lh4bfQgc1_FH765Ew)]

These samples are the same used in our work to compare MSE amongst denoisers.

## Sampling

In addition, you can use the denoisers implemented in this work to draw samples by integrating the PF-ODE:

```
python sample.py --outdir=./tmp --data=PATH_TO_DATASET.zip --denoiser=pspc_sq
```

Which will generate 10 samples to `./tmp/pspc_sq_samples.npy`. Like `denoise.py`, you can specify a set of initial latents with `--z_init`.

```
python sample.py --outdir=./tmp --data=PATH_TO_DATASET.zip --denoiser=pspc_sq  --z_init=path_to_z.npy
```

Will generate 10 samples, starting from the first 10 latents of `path_to_z.npy`. We have uploaded 1000 initial `z` which we used to produce Figure 9 of our work here: [[Link](https://drive.google.com/file/d/1MBKqRL1NEDEYVJ58hfydqSpLKaXfndO6/view?usp=sharing)]

## Citation

```
@article{niedoba2024towards,
  title={Towards a Mechanistic Explanation of Diffusion Model Generalization},
  author={Niedoba, Matthew and Zwartsenberg, Berend and Murphy, Kevin and Wood, Frank},
  journal={arXiv preprint arXiv:2411.19339},
  year={2024}
}
```

## Acknowledgement

This code utilizes portions of the [EDM](https://github.com/NVlabs/edm/) codebase. Addtiionally, `NetworkDenoiser` is
compatible with all network checkpoints from that codebase

## 
