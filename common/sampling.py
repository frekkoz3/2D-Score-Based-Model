from abc import ABC, abstractmethod

import torch
import numpy as np
import math

class Sampler(ABC):

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

class LangevinSampler(Sampler):

    def __init__(
        self,
        score_fn,
        step_size=1e-2,
    ):
        self.score_fn = score_fn
        self.step_size = step_size

    def sample(
    self,
    n_samples,
    n_steps : int = 1000,
    x0=None,
    return_trajectory=False,
    ):

        trajectory = []

        if x0 is None:
            x = torch.randn(n_samples, 2)
        else: 
            x = x0

        if return_trajectory:
            trajectory.append(x.clone())

        for step in range(n_steps):

            with torch.no_grad():

                score = self.score_fn(x)

            noise = torch.randn_like(x)

            x = (
                x
                + self.step_size * score
                + np.sqrt(2 * self.step_size) * noise
            )

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:

            trajectory = torch.stack(trajectory)

            return x, trajectory

        return x

class AnnealedLangevinSampler(Sampler):
    """
    Annealed Langevin Dynamics sampler.

    Given a score function

        score_fn(x, sigma)

    approximating

        nabla log p_sigma(x),

    the sampler progressively decreases the noise level
    following the provided sigma schedule.

    References
    ----------
    Song & Ermon (2019)
    """

    def __init__(
        self,
        score_fn,
        sigmas,
        step_size=1e-4,
        steps_per_sigma=100,
    ):
        self.score_fn = score_fn

        self.sigmas = sigmas # this is a scalar tensor (the network broadcast it)

        self.step_size = step_size
        self.steps_per_sigma = steps_per_sigma

    def sample(
        self,
        x0,
        return_trajectory=False,
    ):

        x = x0.clone()

        trajectory = []
        sigma_history = []

        if return_trajectory:
            trajectory.append(x.clone())

        sigma_min = self.sigmas[-1]

        for sigma in self.sigmas:

            alpha = self.step_size * (sigma / sigma_min) ** 2

            sigma_value = sigma.item()

            for _ in range(self.steps_per_sigma):

                with torch.no_grad():

                    score = self.score_fn(
                        x,
                        sigma,
                    )

                noise = torch.randn_like(x)

                x = (
                    x
                    + alpha * score
                    + math.sqrt(2.0 * alpha) * noise
                )

                if return_trajectory:
                    trajectory.append(x.clone())
                    sigma_history.append(sigma_value)

        history = {
            "sigma": sigma_history,
        }

        if return_trajectory:
            trajectory = torch.stack(trajectory)
            history["trajectory"] = trajectory

        return x, history

if __name__ == '__main__':

    from common.visualization import *
    from common.datasets import *

    distr = GaussianMixture2D(means=((1, 1), (-1, -1)), weights = (100, 1))

    viz = DistributionVisualizer(distr)

    sampler = LangevinSampler(distr.score)

    samples, trajectory = sampler.sample(1000, return_trajectory = True)

    fig, axs = viz.plot_distribution()

    viz.plot_samples(samples, axs)

    plt.show()