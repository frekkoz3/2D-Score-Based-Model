import matplotlib.pyplot as plt
import torch
from common.datasets import *

def plot_distribution(
    distribution : Distribution2D,
    n_samples: int = 5000,
    ax=None,
    figsize=(6, 6),
    alpha=0.5,
    s=5,
):
    """
    Plot samples from a 2D distribution.

    Parameters

    distribution : Distribution2D
        Distribution to visualize.

    n_samples : int
        Number of sampled points.

    ax : matplotlib.axes.Axes, optional
        Existing axes.

    figsize : tuple
        Figure size.

    alpha : float
        Scatter transparency.

    s : float
        Scatter point size.s

    Returns

    fig, ax
    """

    samples = distribution.sample(n_samples)

    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        s=s,
        alpha=alpha,
    )

    if hasattr(distribution, "bounds"):
        xmin, xmax, ymin, ymax = distribution.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    return fig, ax

def make_meshgrid(distribution, resolution=100):
    """
    Creates a regular grid covering the distribution bounds.

    Returns

    xx, yy : meshgrid
    points : Tensor of shape (resolution^2, 2)
    """

    xmin, xmax, ymin, ymax = distribution.bounds

    xs = torch.linspace(xmin, xmax, resolution)
    ys = torch.linspace(ymin, ymax, resolution)

    xx, yy = torch.meshgrid(xs, ys, indexing="xy")

    points = torch.stack(
        [xx.flatten(), yy.flatten()],
        dim=1,
    )

    return xx, yy, points

if __name__ == '__main__':

    distr = GaussianMixture2D(((0, 0), (1, 3)))
    fig, axs = plot_distribution(distr)
    plt.show()