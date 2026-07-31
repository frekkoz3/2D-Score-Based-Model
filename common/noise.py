from abc import ABC, abstractmethod

import torch


class NoiseSchedule(ABC):
    """
    Base class for noise schedules.
    """

    @abstractmethod
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample one noise level for each element of a batch.
        """
        pass

    @property
    @abstractmethod
    def sigmas(self) -> torch.Tensor:
        """
        Return the complete schedule.
        """
        pass

class GeometricNoiseSchedule(NoiseSchedule):
    """
    Geometric sequence of noise levels.

    simga_max → sigma_min
    """

    def __init__(
        self,
        sigma_min=0.01,
        sigma_max=1.0,
        n_levels=10,
        device="cpu",
    ):
        self._sigmas = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(sigma_max)),
                torch.log(torch.tensor(sigma_min)),
                n_levels,
                device=device,
            )
        )

    @property
    def sigmas(self):
        return self._sigmas

    def sample(self, batch_size):

        idx = torch.randint(
            len(self._sigmas),
            (batch_size,),
            device=self._sigmas.device,
        )

        return self._sigmas[idx]

class LinearNoiseSchedule(NoiseSchedule):

    def __init__(
        self,
        sigma_min=0.01,
        sigma_max=1.0,
        n_levels=10,
        device="cpu",
    ):
        self._sigmas = torch.linspace(
            sigma_max,
            sigma_min,
            n_levels,
            device=device,
        )

    @property
    def sigmas(self):
        return self._sigmas

    def sample(self, batch_size):

        idx = torch.randint(
            len(self._sigmas),
            (batch_size,),
            device=self._sigmas.device,
        )

        return self._sigmas[idx]