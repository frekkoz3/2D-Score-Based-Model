import matplotlib.pyplot as plt
import torch

import exp.learning.config as config

from common.datasets import Gaussian2D, GaussianMixture2D
from common.models import NoiseConditionalScoreMLP
from common.visualization import DistributionVisualizer


DEVICE = config.DEVICE


def main():

    dataset = GaussianMixture2D(
        means=config.GAUSSIAN_MEAN,
        std=config.GAUSSIAN_STD,
    )

    model = NoiseConditionalScoreMLP(
        hidden_dim=config.HIDDEN_DIM,
        sigma_embedding_dim=config.SIGMA_EMBEDDING_DIM,
        n_hidden=config.N_HIDDEN,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            f"{config.BASE_PATH}model.pt",
            map_location=DEVICE,
        )
    )

    model.eval()

    visualizer = DistributionVisualizer(dataset)

    fig, ax = plt.subplots(figsize=(6, 6))

    visualizer.plot_samples(
        dataset.sample(5000),
        ax=ax,
    )

    ax.set_title("Training Distribution")

    fig, ax = plt.subplots(figsize=(6, 6))

    visualizer.plot_theoretical_score_field(
        ax=ax
    )

    ax.set_title("True Score Field")

    sigma = torch.full(
        (1,),
        config.SIGMA_MIN,
        device=DEVICE,
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    visualizer.plot_empirical_score_field(
        score_fn=model.to("cpu"),
        sigma=sigma.to("cpu"),
        ax=ax,
    )

    ax.set_title("Learned Score Field")

    plt.show()


if __name__ == "__main__":
    main()