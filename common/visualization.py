from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.datasets import *

class DistributionVisualizer:

    def __init__(
        self,
        distribution : Distribution2D,
        resolution: int = 50,
        figsize=(6, 6),
    ):

        self.distribution = distribution
        self.resolution = resolution
        self.figsize = figsize

        self.xmin, self.xmax, self.ymin, self.ymax = distribution.bounds

        self.xx, self.yy, self.points = self._make_meshgrid()

    def _make_meshgrid(self):

        xs = torch.linspace(
            self.xmin,
            self.xmax,
            self.resolution,
        )

        ys = torch.linspace(
            self.ymin,
            self.ymax,
            self.resolution,
        )

        xx, yy = torch.meshgrid(
            xs,
            ys,
            indexing="xy",
        )

        points = torch.stack(
            [
                xx.flatten(),
                yy.flatten(),
            ],
            dim=1,
        )

        return xx, yy, points

    def _setup_axes(self, ax):

            ax.set_xlim(self.xmin, self.xmax)
            ax.set_ylim(self.ymin, self.ymax)

            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")

            ax.set_aspect("equal")
            ax.grid(alpha=0.3)

    def plot_distribution(
        self,
        n_samples=5000,
        ax=None,
        **kwargs,
    ):

        samples = self.distribution.sample(n_samples)

        if isinstance(samples, torch.Tensor):
            samples = samples.numpy()

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        ax.scatter(
            samples[:, 0],
            samples[:, 1],
            s=5,
            alpha=0.5,
            **kwargs,
        )

        self._setup_axes(ax)

        ax.set_title("Samples")

        return fig, ax

    def plot_density(
        self,
        ax=None,
        cmap="viridis",
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        with torch.no_grad():

            if hasattr(self.distribution, "log_prob"):

                values = self.distribution.log_prob(
                    self.points
                )

            else:

                raise NotImplementedError("The distribution does not provide an interface for the log probability")

        values = values.reshape(
            self.resolution,
            self.resolution,
        )

        im = ax.imshow(
            values.numpy(),
            origin="lower",
            extent=[
                self.xmin,
                self.xmax,
                self.ymin,
                self.ymax,
            ],
            cmap=cmap,
        )

        plt.colorbar(im, ax=ax)

        self._setup_axes(ax)

        ax.set_title("Log density")

        return fig, ax

    def plot_theoretical_score_field(
        self,
        ax=None,
        normalize=False
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        with torch.no_grad():

            if hasattr(self.distribution, "score"):

                score = self.distribution.score(
                    self.points
                )

            else:

                raise NotImplementedError("The distribution does not provide an interface for the score")

        score = score.numpy()
        
        u = score[:, 0].reshape(
            self.resolution,
            self.resolution,
        )

        v = score[:, 1].reshape(
            self.resolution,
            self.resolution,
        )

        if normalize:

            mag = np.sqrt(u**2 + v**2)

            mag = np.maximum(mag, 1e-8)

            u /= mag
            v /= mag

        ax.quiver(
            self.xx,
            self.yy,
            u,
            v,
        )

        self._setup_axes(ax)

        ax.set_title("Score field")

        return fig, ax

    def plot_empirical_score_field(
        self,
        score_fn: Callable,
        ax=None,
        normalize=False
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        with torch.no_grad():
            score = score_fn(self.points)

        score = score.numpy()

        u = score[:, 0].reshape(
            self.resolution,
            self.resolution,
        )

        v = score[:, 1].reshape(
            self.resolution,
            self.resolution,
        )

        if normalize:

            mag = np.sqrt(u**2 + v**2)

            mag = np.maximum(mag, 1e-8)

            u /= mag
            v /= mag

        ax.quiver(
            self.xx,
            self.yy,
            u,
            v,
        )

        self._setup_axes(ax)

        ax.set_title("Score field")

        return fig, ax

if __name__ == '__main__':

    distr = TwoMoons()
    visualizer = DistributionVisualizer(distr)
    fig, axs = visualizer.plot_density()
    plt.show()