from common.datasets import *
import torch

def get_distribution(name: str, **kwargs):

    name = name.lower()

    if name == "gaussian":
        return Gaussian2D(**kwargs)

    if name == "gmm":
        return GaussianMixture2D(**kwargs)

    if name == "moons":
        return TwoMoons(**kwargs)

    if name == "circles":
        return Circles(**kwargs)

    raise ValueError(
        f"Unknown distribution '{name}'"
    )

def perturb(x, sigma):
    
    noise = torch.randn_like(x)

    x_noisy = x + sigma.unsqueeze(-1) * noise

    return x_noisy, noise