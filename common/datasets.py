"""
    This file contains the code for the generation of the datasets.
    More precisely, since this is a 2D explorative project on probability distributions,
    the datasets are indeed 2D distributions.
    The main methods the distributions expose are:
    - sample
    - log_prob(x)
    - score(x)
    - bounds (utility for the visualization)
    The implemented distributions are:
    - 2d gaussian
    - 2d mixture of gaussian (k gaussian, mixing parameter phi)
    - moons
    - circles
    The gaussian family has the theoretical log prob and score implemented.
    The moons and the circles have just the sample methods.
"""
from abc import ABC, abstractmethod

import torch
from sklearn.datasets import make_moons, make_circles

class Distribution2D(ABC):
    """
    Base class for every 2D probability distribution.
    """

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """
        Mandatory.

        Returns tensor of shape (n, 2)
        """
        pass

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optional.

        returns tensor of shape (n, 2) 
        containing log p(x).
        """
        raise NotImplementedError

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optional.

        returns the gradient of log p(x)

        with shape (N,2).
        """
        raise NotImplementedError

    @property
    def bounds(self):
        """
        Utility for the visualization.

        Returns the "distribution's bounds" in probability term
        (excluded the regions where the probability is almost 0).
        """
        raise NotImplementedError

class Gaussian2D(Distribution2D):
    """
    Two-dimensional Gaussian distribution.

    Parameters

    mean : tuple or Tensor of shape (2,)
        Mean vector.

    cov : Tensor of shape (2,2), optional
        Covariance matrix.

    std : float, optional
        Standard deviation. Used only if cov is not provided.
    """

    def __init__(
        self,
        mean=(0.0, 0.0),
        cov=None,
        std=1.0,
    ):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)

        if cov is None:
            cov = (std ** 2) * torch.eye(2)

        self.cov = torch.as_tensor(cov, dtype=torch.float32)

        self.inv_cov = torch.inverse(self.cov)

        self.dist = torch.distributions.MultivariateNormal(
            loc=self.mean,
            covariance_matrix=self.cov,
        )

    def sample(self, n: int) -> torch.Tensor:
        return self.dist.sample((n,))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(x)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        # closed formula for gaussian score
        diff = x - self.mean
        return -(diff @ self.inv_cov.T)

    @property
    def bounds(self):
        radius = 4.0 * torch.sqrt(torch.diag(self.cov))

        xmin = (self.mean[0] - radius[0]).item()
        xmax = (self.mean[0] + radius[0]).item()

        ymin = (self.mean[1] - radius[1]).item()
        ymax = (self.mean[1] + radius[1]).item()

        return xmin, xmax, ymin, ymax

class GaussianMixture2D(Distribution2D):
    """
    Mixture of 2D Gaussians.

    p(x) = sum_i phi_i N(x | mu_i, Sigma_i)

    Parameters
    
        mean : tuple or Tensor of shape (2,)
            Mean vector.
    
        cov : Tensor of shape (2,2), optional
            Covariance matrix.
    
        std : float, optional
            Standard deviation. Used only if cov is not provided.

        weights : float, optional
            Weights vector. This vector will be normalized if it is not.
            Weights will be uniform if not provided.
    """

    def __init__(
        self,
        means,
        covs=None,
        std=0.25,
        weights=None,
    ):
        self.means = torch.as_tensor(means, dtype=torch.float32)

        self.n_components = self.means.shape[0]

        if covs is None:
            covs = [
                (std ** 2) * torch.eye(2)
                for _ in range(self.n_components)
            ]

        self.covs = torch.stack(
            [torch.as_tensor(c, dtype=torch.float32) for c in covs]
        )

        self.inv_covs = torch.inverse(self.covs)

        if weights is None:
            weights = torch.ones(self.n_components)

        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        self.weights /= self.weights.sum()

        self.components = [
            torch.distributions.MultivariateNormal(
                loc=self.means[i],
                covariance_matrix=self.covs[i],
            )
            for i in range(self.n_components)
        ]

    def sample(self, n: int):

        indices = torch.multinomial(
            self.weights,
            n,
            replacement=True,
        )

        samples = []

        for idx in indices:
            samples.append(
                self.components[idx].sample()
            )

        return torch.stack(samples)

    def log_prob(self, x):
        """
            Since p(x) = sum_i phi_i N(x | mu_i, Sigma_i), then 

            log(p(x)) = log(sum_i phi_i N(x | mu_i, Sigma_i))
        """

        log_probs = []

        for w, comp in zip(self.weights, self.components):

            log_probs.append(
                torch.log(w)
                + comp.log_prob(x)
            )

        log_probs = torch.stack(log_probs)

        return torch.logsumexp(log_probs, dim=0)

    def score(self, x):
        """
            Since p(x) = sum_i phi_i N(x | mu_i, Sigma_i), then 

            nabla(log(p(x))) = sum_i r_i nabla(log(p_i(x)))

            where r_i are responsabilities, defined as

            r_i(x) = (phi_i p_i(x)) / (sum_j phi_j p_j(x))

            (p_i(x) is the probability of a datapoint of belonging to 
            the i-th component of the gaussian mixture, which is just
            N(x | m_i, Sigma_i))
        """

        log_probs = []

        scores = []

        for w, comp, mu, inv_cov in zip(
            self.weights,
            self.components,
            self.means,
            self.inv_covs,
        ):

            lp = torch.log(w) + comp.log_prob(x)

            log_probs.append(lp)

            diff = x - mu

            s = -(diff @ inv_cov.T)

            scores.append(s)

        log_probs = torch.stack(log_probs)

        scores = torch.stack(scores)

        responsibilities = torch.softmax(
            log_probs,
            dim=0,
        )

        return (
            responsibilities.unsqueeze(-1)
            * scores
        ).sum(dim=0)

    @property
    def bounds(self):
        margin = 4 * torch.sqrt(
            torch.diagonal(self.covs, dim1=1, dim2=2)
        ).max()

        xmin = self.means[:, 0].min() - margin
        xmax = self.means[:, 0].max() + margin

        ymin = self.means[:, 1].min() - margin
        ymax = self.means[:, 1].max() + margin

        return (
            xmin.item(),
            xmax.item(),
            ymin.item(),
            ymax.item(),
        )

class TwoMoons(Distribution2D):

    def __init__(
        self,
        n_samples=100_000,
        noise=0.05,
        seed=42,
    ):

        x, _ = make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=seed,
        )

        self.data = torch.tensor(
            x,
            dtype=torch.float32,
        )

    def sample(self, n):

        idx = torch.randint(
            0,
            len(self.data),
            (n,),
        )

        return self.data[idx]

    @property
    def bounds(self):
        return -2, 3, -1.5, 2

class Circles(Distribution2D):

    def __init__(
        self,
        n_samples=100_000,
        noise=0.05,
        factor=0.5,
        seed=42,
    ):

        x, _ = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=factor,
            random_state=seed,
        )

        self.data = torch.tensor(
            x,
            dtype=torch.float32,
        )

    def sample(self, n):

        idx = torch.randint(
            0,
            len(self.data),
            (n,),
        )

        return self.data[idx]

    @property
    def bounds(self):
        return -2, 2, -2, 2